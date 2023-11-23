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
    Env, LiveBundleIndex, LiveBundleVec, LiveRangeFlag, LiveRangeIndex, LiveRangeList,
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

        debug_assert!(split_at > bundle_start && split_at < bundle_end);

        // We need to find which LRs fall on each side of the split,
        // which LR we need to split down the middle, then update the
        // current bundle, create a new one, and (re)-queue both.

        trace!(" -> LRs: {:?}", self.bundles[bundle].ranges);

        let mut last_lr_in_old_bundle_idx = 0; // last LR-list index in old bundle
        let mut first_lr_in_new_bundle_idx = 0; // first LR-list index in new bundle
        for (i, entry) in self.bundles[bundle].ranges.iter().enumerate() {
            if split_at > entry.range.from {
                last_lr_in_old_bundle_idx = i;
                first_lr_in_new_bundle_idx = i;
            }
            if split_at < entry.range.to {
                first_lr_in_new_bundle_idx = i;

                // When the bundle contains a fixed constraint, we advance the split point to right
                // before the first instruction with a fixed use present.
                if self.bundles[bundle].cached_fixed() {
                    for u in &self.ranges[entry.index].uses {
                        if u.pos < split_at {
                            continue;
                        }

                        if matches!(u.operand.constraint(), OperandConstraint::FixedReg { .. }) {
                            split_at = ProgPoint::before(u.pos.inst());

                            if split_at > entry.range.from {
                                last_lr_in_old_bundle_idx = i;
                            }

                            trace!(" -> advancing split point to {split_at:?}");

                            trim_ends_into_spill_bundle = false;

                            break;
                        }
                    }
                }

                break;
            }
        }

        trace!(
            " -> last LR in old bundle: LR {:?}",
            self.bundles[bundle].ranges[last_lr_in_old_bundle_idx]
        );
        trace!(
            " -> first LR in new bundle: LR {:?}",
            self.bundles[bundle].ranges[first_lr_in_new_bundle_idx]
        );

        // Take the sublist of LRs that will go in the new bundle.
        let mut new_lr_list: LiveRangeList = self.bundles[bundle]
            .ranges
            .iter()
            .cloned()
            .skip(first_lr_in_new_bundle_idx)
            .collect();
        self.bundles[bundle]
            .ranges
            .truncate(last_lr_in_old_bundle_idx + 1);
        self.bundles[bundle].ranges.shrink_to_fit();

        // If the first entry in `new_lr_list` is a LR that is split
        // down the middle, replace it with a new LR and chop off the
        // end of the same LR in the original list.
        if split_at > new_lr_list[0].range.from {
            debug_assert_eq!(last_lr_in_old_bundle_idx, first_lr_in_new_bundle_idx);
            let orig_lr = new_lr_list[0].index;
            let new_lr = self.ranges.add(CodeRange {
                from: split_at,
                to: new_lr_list[0].range.to,
            });
            self.ranges[new_lr].vreg = self.ranges[orig_lr].vreg;
            trace!(" -> splitting LR {:?} into {:?}", orig_lr, new_lr);
            let first_use = self.ranges[orig_lr]
                .uses
                .iter()
                .position(|u| u.pos >= split_at)
                .unwrap_or(self.ranges[orig_lr].uses.len());
            let rest_uses: UseList = self.ranges[orig_lr]
                .uses
                .iter()
                .cloned()
                .skip(first_use)
                .collect();
            self.ranges[new_lr].uses = rest_uses;
            self.ranges[orig_lr].uses.truncate(first_use);
            self.ranges[orig_lr].uses.shrink_to_fit();
            self.recompute_range_properties(orig_lr);
            self.recompute_range_properties(new_lr);
            new_lr_list[0].index = new_lr;
            new_lr_list[0].range = self.ranges[new_lr].range;
            self.ranges[orig_lr].range.to = split_at;
            self.bundles[bundle].ranges[last_lr_in_old_bundle_idx].range =
                self.ranges[orig_lr].range;

            // Perform a lazy split in the VReg data. We just
            // append the new LR and its range; we will sort by
            // start of range, and fix up range ends, once when we
            // iterate over the VReg's ranges after allocation
            // completes (this is the only time when order
            // matters).
            self.vregs[self.ranges[new_lr].vreg]
                .ranges
                .push(LiveRangeListEntry {
                    range: self.ranges[new_lr].range,
                    index: new_lr,
                });
        }

        let new_bundle = self.bundles.add();
        trace!(" -> creating new bundle {:?}", new_bundle);
        self.bundles[new_bundle].spillset = spillset;
        for entry in &new_lr_list {
            self.ranges[entry.index].bundle = new_bundle;
        }
        self.bundles[new_bundle].ranges = new_lr_list;

        if trim_ends_into_spill_bundle {
            // Finally, handle moving LRs to the spill bundle when
            // appropriate: If the first range in `new_bundle` or last
            // range in `bundle` has "empty space" beyond the first or
            // last use (respectively), trim it and put an empty LR into
            // the spill bundle.  (We are careful to treat the "starts at
            // def" flag as an implicit first def even if no def-type Use
            // is present.)
            while let Some(entry) = self.bundles[bundle].ranges.last().cloned() {
                let end = entry.range.to;
                let vreg = self.ranges[entry.index].vreg;
                let last_use = self.ranges[entry.index].uses.last().map(|u| u.pos);
                if last_use.is_none() {
                    let spill = self
                        .get_or_create_spill_bundle(bundle, /* create_if_absent = */ true)
                        .unwrap();
                    trace!(
                        " -> bundle {:?} range {:?}: no uses; moving to spill bundle {:?}",
                        bundle,
                        entry.index,
                        spill
                    );
                    self.bundles[spill].ranges.push(entry);
                    self.bundles[bundle].ranges.pop();
                    self.ranges[entry.index].bundle = spill;
                    continue;
                }
                let last_use = last_use.unwrap();
                let split = ProgPoint::before(last_use.inst().next());
                if split < end {
                    let spill = self
                        .get_or_create_spill_bundle(bundle, /* create_if_absent = */ true)
                        .unwrap();
                    self.bundles[bundle].ranges.last_mut().unwrap().range.to = split;
                    self.ranges[self.bundles[bundle].ranges.last().unwrap().index]
                        .range
                        .to = split;
                    let range = CodeRange {
                        from: split,
                        to: end,
                    };
                    let empty_lr = self.ranges.add(range);
                    self.bundles[spill].ranges.push(LiveRangeListEntry {
                        range,
                        index: empty_lr,
                    });
                    self.ranges[empty_lr].bundle = spill;
                    self.vregs[vreg].ranges.push(LiveRangeListEntry {
                        range,
                        index: empty_lr,
                    });
                    trace!(
                        " -> bundle {:?} range {:?}: last use implies split point {:?}",
                        bundle,
                        entry.index,
                        split
                    );
                    trace!(
                    " -> moving trailing empty region to new spill bundle {:?} with new LR {:?}",
                    spill,
                    empty_lr
                );
                }
                break;
            }
            while let Some(entry) = self.bundles[new_bundle].ranges.first().cloned() {
                if self.ranges[entry.index].has_flag(LiveRangeFlag::StartsAtDef) {
                    break;
                }
                let start = entry.range.from;
                let vreg = self.ranges[entry.index].vreg;
                let first_use = self.ranges[entry.index].uses.first().map(|u| u.pos);
                if first_use.is_none() {
                    let spill = self
                        .get_or_create_spill_bundle(new_bundle, /* create_if_absent = */ true)
                        .unwrap();
                    trace!(
                        " -> bundle {:?} range {:?}: no uses; moving to spill bundle {:?}",
                        new_bundle,
                        entry.index,
                        spill
                    );
                    self.bundles[spill].ranges.push(entry);
                    self.bundles[new_bundle].ranges.drain(..1);
                    self.ranges[entry.index].bundle = spill;
                    continue;
                }
                let first_use = first_use.unwrap();
                let split = ProgPoint::before(first_use.inst());
                if split > start {
                    let spill = self
                        .get_or_create_spill_bundle(new_bundle, /* create_if_absent = */ true)
                        .unwrap();
                    self.bundles[new_bundle]
                        .ranges
                        .first_mut()
                        .unwrap()
                        .range
                        .from = split;
                    self.ranges[self.bundles[new_bundle].ranges.first().unwrap().index]
                        .range
                        .from = split;
                    let range = CodeRange {
                        from: start,
                        to: split,
                    };
                    let empty_lr = self.ranges.add(range);
                    self.bundles[spill].ranges.push(LiveRangeListEntry {
                        range,
                        index: empty_lr,
                    });
                    self.ranges[empty_lr].bundle = spill;
                    self.vregs[vreg].ranges.push(LiveRangeListEntry {
                        range,
                        index: empty_lr,
                    });
                    trace!(
                        " -> bundle {:?} range {:?}: first use implies split point {:?}",
                        bundle,
                        entry.index,
                        first_use,
                    );
                    trace!(
                        " -> moving leading empty region to new spill bundle {:?} with new LR {:?}",
                        spill,
                        empty_lr
                    );
                }
                break;
            }
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
                if u.operand.constraint() == OperandConstraint::Any {
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
}
