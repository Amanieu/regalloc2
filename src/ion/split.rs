/*
 * This file was initially derived from the files
 * `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox, and was
 * originally licensed under the Mozilla Public License 2.0. We
 * subsequently relicensed it to Apache-2.0 WITH LLVM-exception (see
 * https://github.com/bytecodealliance/regalloc2/issues/7).
 *
 * Since the initial port, the design has been substantially evolved
 * and optimized.
 */

//! Code related to bundle splitting.

use super::{
    requirement::Requirement, Env, LiveBundleIndex, LiveBundleVec, LiveRangeFlag, LiveRangeIndex,
    LiveRangeListEntry, UseList, VRegIndex,
};
use crate::{
    ion::data_structures::{CodeRange, MAX_SPLITS_PER_SPILLSET},
    Allocation, Function, FxHashSet, Inst, InstPosition, OperandConstraint, OperandKind, PReg,
    ProgPoint,
};
use core::fmt::Debug;
use smallvec::{smallvec, SmallVec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocRegResult {
    Allocated(Allocation),
    Conflict(LiveBundleVec, ProgPoint),
    ConflictWithFixed(u32, ProgPoint),
    ConflictHighCost,
}

impl<'a, F: Function> Env<'a, F> {
    pub fn split_and_requeue_bundle(
        &mut self,
        bundle: LiveBundleIndex,
        mut split_at: ProgPoint,
        reg_hint: PReg,
        // Do we trim the parts around the split and put them in the
        // spill bundle?
        mut trim_ends_into_spill_bundle: bool,
    ) {
        self.stats.splits += 1;
        trace!(
            "split bundle {:?} at {:?} and requeue with reg hint (for first part) {:?}",
            bundle,
            split_at,
            reg_hint,
        );

        // Split `bundle` at `split_at`, creating new LiveRanges and
        // bundles (and updating vregs' linked lists appropriately),
        // and enqueue the new bundles.

        let spillset = self.bundles[bundle].spillset;

        // Have we reached the maximum split count? If so, fall back
        // to a "minimal bundles and spill bundle" setup for this
        // bundle. See the doc-comment on
        // `split_into_minimal_bundles()` above for more.
        if self.spillsets[spillset].splits >= MAX_SPLITS_PER_SPILLSET {
            self.split_into_minimal_bundles(bundle, reg_hint);
            return;
        }
        self.spillsets[spillset].splits += 1;

        debug_assert!(!self.bundles[bundle].ranges.is_empty());
        // Split point *at* start is OK; this means we peel off
        // exactly one use to create a minimal bundle.
        let bundle_start = self.bundles[bundle].ranges.first().unwrap().range.from;
        debug_assert!(split_at >= bundle_start);
        let bundle_end = self.bundles[bundle].ranges.last().unwrap().range.to;
        debug_assert!(split_at < bundle_end);

        // Is the split point *at* the start? If so, peel off the
        // first use: set the split point just after it, or just
        // before it if it comes after the start of the bundle.
        if split_at == bundle_start {
            // Find any uses; if none, just chop off one instruction.
            let mut first_use = None;
            'outer: for entry in &self.bundles[bundle].ranges {
                for u in &self.ranges[entry.index].uses {
                    first_use = Some(u.pos);
                    break 'outer;
                }
            }
            trace!(" -> first use loc is {:?}", first_use);
            split_at = match first_use {
                Some(pos) => {
                    if pos.inst() == bundle_start.inst() {
                        ProgPoint::before(pos.inst().next())
                    } else {
                        ProgPoint::before(pos.inst())
                    }
                }
                None => ProgPoint::before(
                    self.bundles[bundle]
                        .ranges
                        .first()
                        .unwrap()
                        .range
                        .from
                        .inst()
                        .next(),
                ),
            };
            trace!(
                "split point is at bundle start; advancing to {:?}",
                split_at
            );
        } else {
            // Don't split in the middle of an instruction -- this could
            // create impossible moves (we cannot insert a move between an
            // instruction's uses and defs).
            if split_at.pos() == InstPosition::After {
                split_at = split_at.next();
            }
            if split_at >= bundle_end {
                split_at = split_at.prev().prev();
            }
        }

        // If this bundle contains a fixed constraint and the first fixed use is
        // located after the split point then move the split point forward to
        // just before the first fixed use.
        //
        // This allows the first half of the bundle to be fully unconstrained by
        // the fixed use, which may make it allocatable.
        if self.bundles[bundle].cached_fixed() {
            let first_fixed_use = self.bundles[bundle]
                .ranges
                .iter()
                .flat_map(|entry| self.ranges[entry.index].uses.iter())
                .filter(|u| matches!(u.operand.constraint(), OperandConstraint::FixedReg(_)))
                .next()
                .unwrap();
            if first_fixed_use.pos > split_at {
                let before_fixed = ProgPoint::before(first_fixed_use.pos.inst());
                trace!(" -> advancing split point to first fixed use at {before_fixed:?}");
                split_at = self.adjust_split_point_backward(before_fixed, split_at);
                trim_ends_into_spill_bundle = false;
            }
        }

        let new_bundle = self.bundles.add();
        trace!(" -> creating new bundle {:?}", new_bundle);
        self.bundles[new_bundle].spillset = spillset;
        self.split_bundle_at(bundle, new_bundle, split_at, true);

        if trim_ends_into_spill_bundle {
            // Finally, handle moving LRs to the spill bundle when
            // appropriate: If the first range in `new_bundle` or last
            // range in `bundle` has "empty space" beyond the first or
            // last use (respectively), trim it and put an empty LR into
            // the spill bundle.  (We are careful to treat the "starts at
            // def" flag as an implicit first def even if no def-type Use
            // is present.)
            self.trim_bundle_tail(bundle);
            self.trim_bundle_head(new_bundle);
        }

        if self.bundles[bundle].ranges.len() > 0 {
            self.recompute_bundle_properties(bundle);
            let prio = self.bundles[bundle].prio;
            self.allocation_queue
                .insert(bundle, prio as usize, reg_hint);
        }
        if self.bundles[new_bundle].ranges.len() > 0 {
            self.recompute_bundle_properties(new_bundle);
            let prio = self.bundles[new_bundle].prio;
            self.allocation_queue
                .insert(new_bundle, prio as usize, reg_hint);
        }
    }

    /// Core functionality for splitting a bundle into two halves and moving the
    /// first or second half into another bundle.
    ///
    /// Note that this doesn't call `recompute_bundle_properties`, it is the
    /// caller's responsibility to do so if necessary.
    fn split_bundle_at(
        &mut self,
        bundle_idx: LiveBundleIndex,
        dest_bundle_idx: LiveBundleIndex,
        split_at: ProgPoint,
        move_second_half: bool,
    ) {
        let (src_bundle, dest_bundle) = self.bundles.get_pair_mut(bundle_idx, dest_bundle_idx);
        let existing_dest_bundle_ranges = dest_bundle.ranges.len();

        // Splits must happen at the before position of an instruction.
        debug_assert_eq!(split_at.pos(), InstPosition::Before);

        // The split point must be within the bundle.
        debug_assert!(split_at > src_bundle.ranges[0].range.from);
        debug_assert!(split_at < src_bundle.ranges.last().unwrap().range.to);

        // We need to find which LRs fall on each side of the split,
        // and which LR we need to split down the middle.

        trace!(
            "splitting bundle {:?} at {:?}, moving {} half into bundle {:?}",
            bundle_idx,
            split_at,
            if move_second_half { "second" } else { "first" },
            dest_bundle_idx
        );
        trace!(" -> LRs: {:?}", src_bundle.ranges);

        let first_lr_in_second_half = src_bundle
            .ranges
            .partition_point(|r| r.range.to <= split_at);
        let last_lr_in_first_half =
            if src_bundle.ranges[first_lr_in_second_half].range.from < split_at {
                first_lr_in_second_half
            } else {
                first_lr_in_second_half - 1
            };

        trace!(
            " -> last LR in first half: LR {:?}",
            src_bundle.ranges[last_lr_in_first_half]
        );
        trace!(
            " -> first LR in second half: LR {:?}",
            src_bundle.ranges[first_lr_in_second_half]
        );

        let lr_to_split = src_bundle.ranges[last_lr_in_first_half].clone();
        if move_second_half {
            trace!(" -> moving second half into bundle {:?}", dest_bundle_idx);
            dest_bundle
                .ranges
                .extend_from_slice(&src_bundle.ranges[first_lr_in_second_half..]);
            src_bundle.ranges.truncate(last_lr_in_first_half + 1);
        } else {
            trace!(" -> moving first half into bundle {:?}", dest_bundle_idx);
            dest_bundle
                .ranges
                .extend_from_slice(&src_bundle.ranges[..last_lr_in_first_half + 1]);
            src_bundle.ranges.drain(..first_lr_in_second_half);
        }
        src_bundle.ranges.shrink_to_fit();

        // If both bundles share a LR that is split down the middle, split it
        // into 2 separate LRs.
        if last_lr_in_first_half == first_lr_in_second_half {
            let orig_lr_index = lr_to_split.index;
            let new_lr_index = self.ranges.add(CodeRange {
                from: split_at,
                to: lr_to_split.range.to,
            });
            let (orig_lr, new_lr) = self.ranges.get_pair_mut(orig_lr_index, new_lr_index);
            orig_lr.range.to = split_at;
            new_lr.vreg = orig_lr.vreg;
            trace!(
                " -> splitting LR {:?} into {:?}",
                orig_lr_index,
                new_lr_index
            );

            // Transfer all uses after the split point to the new live range.
            let split_use = orig_lr.uses.partition_point(|u| u.pos < split_at);
            new_lr.uses = UseList::from_slice(&orig_lr.uses[split_use..]);
            orig_lr.uses.truncate(split_use);
            orig_lr.uses.shrink_to_fit();

            if move_second_half {
                dest_bundle.ranges[existing_dest_bundle_ranges].range.from = split_at;
                dest_bundle.ranges[existing_dest_bundle_ranges].index = new_lr_index;
                src_bundle.ranges[last_lr_in_first_half].range.to = split_at;
            } else {
                dest_bundle.ranges[existing_dest_bundle_ranges + last_lr_in_first_half]
                    .range
                    .to = split_at;
                src_bundle.ranges[0].range.from = split_at;
                src_bundle.ranges[0].index = new_lr_index;
                new_lr.bundle = bundle_idx;
            }

            // Perform a lazy split in the VReg data. We just
            // append the new LR and its range; we will sort by
            // start of range, and fix up range ends, once when we
            // iterate over the VReg's ranges after allocation
            // completes (this is the only time when order
            // matters).
            self.vregs[new_lr.vreg].ranges.push(LiveRangeListEntry {
                range: new_lr.range,
                index: new_lr_index,
            });

            self.recompute_range_properties(orig_lr_index);
            self.recompute_range_properties(new_lr_index);
        }

        for entry in &self.bundles[dest_bundle_idx].ranges[existing_dest_bundle_ranges..] {
            self.ranges[entry.index].bundle = dest_bundle_idx;
        }
    }

    /// Trims any "empty space" at the end of the given bundle and moves the
    /// live ranges into the spill bundle.
    fn trim_bundle_tail(&mut self, bundle: LiveBundleIndex) {
        // Select a split point after the last use.
        let Some(last_use) = self.bundles[bundle]
            .ranges
            .iter()
            .flat_map(|entry| self.ranges[entry.index].uses.iter())
            // Trim AnyCold uses in addition to empty live ranges: these have no
            // spill weight so there is no cost to moving them to the spill
            // bundle.
            .filter(|u| {
                !(u.operand.constraint() == OperandConstraint::AnyCold
                    && u.operand.kind() == OperandKind::Use)
            })
            .next_back()
        else {
            return;
        };
        let bundle_end = ProgPoint::before(
            self.bundles[bundle]
                .ranges
                .last()
                .unwrap()
                .range
                .to
                .next()
                .inst(),
        );
        let split = ProgPoint::before(last_use.pos.inst().next());
        let split = self.adjust_split_point_forward(split, bundle_end);

        if split != bundle_end {
            let spill = self
                .get_or_create_spill_bundle(bundle, /* create_if_absent = */ true)
                .unwrap();
            trace!(
                "trimming tail of bundle {:?} into spill bundle {:?}",
                bundle,
                spill
            );
            self.split_bundle_at(bundle, spill, split, true);
        }
    }

    /// Trims any "empty space" at the start of the given bundle and moves the
    /// live ranges into the spill bundle.
    fn trim_bundle_head(&mut self, bundle: LiveBundleIndex) {
        // Select a split point before the first use.
        let Some(first_use) = self.bundles[bundle]
            .ranges
            .iter()
            .flat_map(|entry| self.ranges[entry.index].uses.iter())
            // Trim AnyCold uses in addition to empty live ranges: these have no
            // spill weight so there is no cost to moving them to the spill
            // bundle.
            .filter(|u| {
                !(u.operand.constraint() == OperandConstraint::AnyCold
                    && u.operand.kind() == OperandKind::Use)
            })
            .next()
        else {
            return;
        };
        let bundle_start = ProgPoint::before(
            self.bundles[bundle]
                .ranges
                .first()
                .unwrap()
                .range
                .from
                .inst(),
        );
        let split = ProgPoint::before(first_use.pos.inst());
        let split = self.adjust_split_point_backward(split, bundle_start);

        if split != bundle_start {
            let spill = self
                .get_or_create_spill_bundle(bundle, /* create_if_absent = */ true)
                .unwrap();
            trace!(
                "trimming head of bundle {:?} into spill bundle {:?}",
                bundle,
                spill
            );
            self.split_bundle_at(bundle, spill, split, false);
        }
    }

    /// Splits the given bundle into minimal bundles per Use, falling
    /// back onto the spill bundle. This must work for any bundle no
    /// matter how many conflicts.
    ///
    /// This is meant to solve a quadratic-cost problem that exists
    /// with "normal" splitting as implemented above. With that
    /// procedure, , splitting a bundle produces two
    /// halves. Furthermore, it has cost linear in the length of the
    /// bundle, because the resulting half-bundles have their
    /// requirements recomputed with a new scan, and because we copy
    /// half the use-list over to the tail end sub-bundle.
    ///
    /// This works fine when a bundle has a handful of splits overall,
    /// but not when an input has a systematic pattern of conflicts
    /// that will require O(|bundle|) splits (e.g., every Use is
    /// constrained to a different fixed register than the last
    /// one). In such a case, we get quadratic behavior.
    ///
    /// This method implements a direct split into minimal bundles
    /// along the whole length of the bundle, putting the regions
    /// without uses in the spill bundle. We do this once the number
    /// of splits in an original bundle (tracked by spillset) reaches
    /// a pre-determined limit.
    ///
    /// This basically approximates what a non-splitting allocator
    /// would do: it "spills" the whole bundle to possibly a
    /// stackslot, or a second-chance register allocation at best, via
    /// the spill bundle; and then does minimal reservations of
    /// registers just at uses/defs and moves the "spilled" value
    /// into/out of them immediately.
    pub fn split_into_minimal_bundles(&mut self, bundle: LiveBundleIndex, reg_hint: PReg) {
        let mut removed_lrs: FxHashSet<LiveRangeIndex> = FxHashSet::default();
        let mut removed_lrs_vregs: FxHashSet<VRegIndex> = FxHashSet::default();
        let mut new_lrs: SmallVec<[(VRegIndex, LiveRangeIndex); 16]> = smallvec![];
        let mut new_bundles: SmallVec<[LiveBundleIndex; 16]> = smallvec![];

        let spillset = self.bundles[bundle].spillset;
        let spill = self
            .get_or_create_spill_bundle(bundle, /* create_if_absent = */ true)
            .unwrap();

        trace!(
            "Splitting bundle {:?} into minimal bundles with reg hint {}",
            bundle,
            reg_hint
        );

        let mut last_lr: Option<LiveRangeIndex> = None;
        let mut last_bundle: Option<LiveBundleIndex> = None;
        let mut last_inst: Option<Inst> = None;
        let mut last_vreg: Option<VRegIndex> = None;

        let mut spill_uses = UseList::new();

        for entry in core::mem::take(&mut self.bundles[bundle].ranges) {
            let lr_from = entry.range.from;
            let lr_to = entry.range.to;
            let vreg = self.ranges[entry.index].vreg;

            removed_lrs.insert(entry.index);
            removed_lrs_vregs.insert(vreg);
            trace!(" -> removing old LR {:?} for vreg {:?}", entry.index, vreg);

            let mut spill_range = entry.range;
            let mut spill_starts_def = false;

            let mut last_live_pos = entry.range.from;
            for u in core::mem::take(&mut self.ranges[entry.index].uses) {
                trace!("   -> use {:?} (last_live_pos {:?})", u, last_live_pos);

                let is_def = u.operand.kind() == OperandKind::Def;

                // If this use has an `any` constraint, eagerly migrate it to the spill range. The
                // reasoning here is that in the second-chance allocation for the spill bundle,
                // any-constrained uses will be easy to satisfy. Solving those constraints earlier
                // could create unnecessary conflicts with existing bundles that need to fit in a
                // register, more strict requirements, so we delay them eagerly.
                if matches!(
                    u.operand.constraint(),
                    OperandConstraint::Any | OperandConstraint::AnyCold
                ) {
                    trace!("    -> migrating this any-constrained use to the spill range");
                    spill_uses.push(u);

                    // Remember if we're moving the def of this vreg into the spill range, so that
                    // we can set the appropriate flags on it later.
                    spill_starts_def = spill_starts_def || is_def;

                    continue;
                }

                // If this is a def of the vreg the entry cares about, make sure that the spill
                // range starts right before the next instruction so that the value is available.
                if is_def {
                    trace!("    -> moving the spill range forward by one");
                    spill_range.from = ProgPoint::before(u.pos.inst().next());
                }

                // If we just created a LR for this inst at the last
                // pos, add this use to the same LR.
                if Some(u.pos.inst()) == last_inst && Some(vreg) == last_vreg {
                    self.ranges[last_lr.unwrap()].uses.push(u);
                    trace!("    -> appended to last LR {:?}", last_lr.unwrap());
                    continue;
                }

                // The minimal bundle runs through the whole inst
                // (up to the Before of the next inst), *unless*
                // the original LR was only over the Before (up to
                // the After) of this inst.
                let to = core::cmp::min(ProgPoint::before(u.pos.inst().next()), lr_to);

                // If the last bundle was at the same inst, add a new
                // LR to the same bundle; otherwise, create a LR and a
                // new bundle.
                if Some(u.pos.inst()) == last_inst {
                    let cr = CodeRange { from: u.pos, to };
                    let lr = self.ranges.add(cr);
                    new_lrs.push((vreg, lr));
                    self.ranges[lr].uses.push(u);
                    self.ranges[lr].vreg = vreg;

                    trace!(
                        "    -> created new LR {:?} but adding to existing bundle {:?}",
                        lr,
                        last_bundle.unwrap()
                    );
                    // Edit the previous LR to end mid-inst.
                    self.bundles[last_bundle.unwrap()]
                        .ranges
                        .last_mut()
                        .unwrap()
                        .range
                        .to = u.pos;
                    self.ranges[last_lr.unwrap()].range.to = u.pos;
                    // Add this LR to the bundle.
                    self.bundles[last_bundle.unwrap()]
                        .ranges
                        .push(LiveRangeListEntry {
                            range: cr,
                            index: lr,
                        });
                    self.ranges[lr].bundle = last_bundle.unwrap();
                    last_live_pos = ProgPoint::before(u.pos.inst().next());
                    continue;
                }

                // Otherwise, create a new LR.
                let pos = ProgPoint::before(u.pos.inst());
                let pos = core::cmp::max(lr_from, pos);
                let cr = CodeRange { from: pos, to };
                let lr = self.ranges.add(cr);
                new_lrs.push((vreg, lr));
                self.ranges[lr].uses.push(u);
                self.ranges[lr].vreg = vreg;

                // Create a new bundle that contains only this LR.
                let new_bundle = self.bundles.add();
                self.ranges[lr].bundle = new_bundle;
                self.bundles[new_bundle].spillset = spillset;
                self.bundles[new_bundle].ranges.push(LiveRangeListEntry {
                    range: cr,
                    index: lr,
                });
                new_bundles.push(new_bundle);

                // If this use was a Def, set the StartsAtDef flag for the new LR.
                if is_def {
                    self.ranges[lr].set_flag(LiveRangeFlag::StartsAtDef);
                }

                trace!(
                    "    -> created new LR {:?} range {:?} with new bundle {:?} for this use",
                    lr,
                    cr,
                    new_bundle
                );

                last_live_pos = ProgPoint::before(u.pos.inst().next());

                last_lr = Some(lr);
                last_bundle = Some(new_bundle);
                last_inst = Some(u.pos.inst());
                last_vreg = Some(vreg);
            }

            if !spill_range.is_empty() {
                // Make one entry in the spill bundle that covers the whole range.
                // TODO: it might be worth tracking enough state to only create this LR when there is
                // open space in the original LR.
                let spill_lr = self.ranges.add(spill_range);
                self.ranges[spill_lr].vreg = vreg;
                self.ranges[spill_lr].bundle = spill;
                self.ranges[spill_lr].uses.extend(spill_uses.drain(..));
                new_lrs.push((vreg, spill_lr));

                if spill_starts_def {
                    self.ranges[spill_lr].set_flag(LiveRangeFlag::StartsAtDef);
                }

                self.bundles[spill].ranges.push(LiveRangeListEntry {
                    range: spill_range,
                    index: spill_lr,
                });
                self.ranges[spill_lr].bundle = spill;
                trace!(
                    "  -> added spill range {:?} in new LR {:?} in spill bundle {:?}",
                    spill_range,
                    spill_lr,
                    spill
                );
            } else {
                assert!(spill_uses.is_empty());
            }
        }

        // Remove all of the removed LRs from respective vregs' lists.
        for vreg in removed_lrs_vregs {
            self.vregs[vreg]
                .ranges
                .retain(|entry| !removed_lrs.contains(&entry.index));
        }

        // Add the new LRs to their respective vreg lists.
        for (vreg, lr) in new_lrs {
            let range = self.ranges[lr].range;
            let entry = LiveRangeListEntry { range, index: lr };
            self.vregs[vreg].ranges.push(entry);
        }

        // Recompute bundle properties for all new bundles and enqueue
        // them.
        for bundle in new_bundles {
            if self.bundles[bundle].ranges.len() > 0 {
                self.recompute_bundle_properties(bundle);
                let prio = self.bundles[bundle].prio;
                self.allocation_queue
                    .insert(bundle, prio as usize, reg_hint);
            }
        }
    }

    /// If the given split point is within a loop, try to move it out of the
    /// loop by moving it forward, up to the given limit.
    pub fn adjust_split_point_forward(&self, mut split: ProgPoint, limit: ProgPoint) -> ProgPoint {
        debug_assert!(split <= limit);
        let orig_split = split;
        while split != limit {
            let block = self.cfginfo.insn_block[split.inst().index()];
            let next_outer_block = self.cfginfo.next_outer_loop[block.index()];
            if next_outer_block.is_invalid() {
                break;
            }
            debug_assert!(
                self.cfginfo.approx_loop_depth[next_outer_block.index()]
                    < self.cfginfo.approx_loop_depth[block.index()]
            );

            let new_split = self.cfginfo.block_entry[next_outer_block.index()];
            if new_split <= limit {
                split = new_split;
            } else {
                break;
            }
        }
        if split != orig_split {
            trace!(" -> moving split point out of loop from {orig_split:?} to {split:?}");
        }
        split
    }

    /// If the given split point is within a loop, try to move it out of the
    /// loop by moving it backward, up to the given limit.
    pub fn adjust_split_point_backward(&self, mut split: ProgPoint, limit: ProgPoint) -> ProgPoint {
        debug_assert!(split >= limit);
        let orig_split = split;
        while split != limit {
            let block = self.cfginfo.insn_block[split.inst().index()];
            let prev_outer_block = self.cfginfo.prev_outer_loop[block.index()];
            if prev_outer_block.is_invalid() {
                break;
            }
            debug_assert!(
                self.cfginfo.approx_loop_depth[prev_outer_block.index()]
                    < self.cfginfo.approx_loop_depth[block.index()]
            );

            // Place the split before the terminator instruction of the block.
            let new_split =
                ProgPoint::before(self.cfginfo.block_exit[prev_outer_block.index()].inst());
            if new_split >= limit {
                split = new_split;
            } else {
                break;
            }
        }
        if split != orig_split {
            trace!(" -> moving split point out of loop from {orig_split:?} to {split:?}");
        }
        split
    }

    /// When bundles are first created, they may have conflicting requirements,
    /// for example two uses with different fixed registers. Additionally,
    /// bundle merging may have introduced additional conflict.
    ///
    /// We handle all requirement conflicts upfront by splitting the bundle into
    /// pieces which each have satisfiable requirements.
    pub fn legalize_bundle_requirements(&mut self) {
        // New bundles may be inserted while we iterate, but those are
        // guaranteed to have satisfiable requirements.
        for bundle in 0..self.bundles.len() {
            self.split_bundle_for_requirements(LiveBundleIndex::new(bundle));
        }
    }

    fn split_bundle_for_requirements(&mut self, bundle: LiveBundleIndex) {
        // Nothing to do if the bundle has no fixed or stack constraints
        if !self.bundles[bundle].cached_fixed() && !self.bundles[bundle].cached_stack() {
            return;
        }

        // Outer loop for each time we need to split the bundle.
        'outer: loop {
            trace!("Checking bundle {bundle:?} for conflicting requirements");

            // Iterate over the uses in the bundle in reverse order.
            let mut last_fixed_reg = None;
            let mut last_fixed_stack = None;
            let mut last_stack = None;
            let mut last_reg = None;
            let mut requirement = Requirement::Any;
            for u in self.bundles[bundle]
                .ranges
                .iter()
                .flat_map(|entry| self.ranges[entry.index].uses.iter())
                .rev()
            {
                trace!("  -> use {:?}", u);

                // Check whether the new requirement from the use conflicts with
                // existing requirements.
                let new_req = self.requirement_from_operand(u.operand);
                match new_req.merge(requirement) {
                    Ok(merged) => {
                        // No conflict! Keep track of the last seen use of each
                        // type and continue.
                        requirement = merged;
                        trace!("     -> req {requirement:?}");
                        match new_req {
                            Requirement::FixedReg(_) => last_fixed_reg = Some(u.pos),
                            Requirement::FixedStack(_) => last_fixed_stack = Some(u.pos),
                            Requirement::Register => last_reg = Some(u.pos),
                            Requirement::Stack => last_stack = Some(u.pos),
                            Requirement::Any => {}
                        }
                    }
                    Err(_) => {
                        trace!("     -> conflict");

                        // Conflict! We need to split the bundle to make it
                        // allocatable. Find the previous conflicting use.
                        let potentially_conflicting = match new_req {
                            Requirement::FixedReg(_) => {
                                [last_fixed_reg, last_fixed_stack, last_stack]
                            }
                            Requirement::FixedStack(_) => {
                                [last_fixed_reg, last_fixed_stack, last_reg]
                            }
                            Requirement::Register => [last_fixed_stack, last_stack, None],
                            Requirement::Stack => [last_fixed_reg, last_fixed_stack, last_reg],
                            Requirement::Any => unreachable!(),
                        };
                        let mut conflicting = None;
                        for pos in potentially_conflicting {
                            let Some(pos) = pos else { continue };
                            match conflicting {
                                Some(prev) => conflicting = Some(core::cmp::min(prev, pos)),
                                None => conflicting = Some(pos),
                            }
                        }
                        let conflicting = conflicting.unwrap();

                        // Split the bundle between the 2 conflicting uses,
                        // while taking loop depth into account.
                        let start = ProgPoint::before(u.pos.inst().next());
                        let end = ProgPoint::before(conflicting.inst());
                        let split_at = self.adjust_split_point_backward(end, start);
                        let new_bundle = self.bundles.add();
                        trace!(" -> creating new bundle {:?}", new_bundle);
                        self.bundles[new_bundle].spillset = self.bundles[bundle].spillset;
                        self.split_bundle_at(bundle, new_bundle, split_at, true);

                        // Now that we've split off a conflicting use, continue
                        // the scan in the original bundle.
                        continue 'outer;
                    }
                }
            }

            // No conflicts found, we're done!
            break;
        }
    }
}
