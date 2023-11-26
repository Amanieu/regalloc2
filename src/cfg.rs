/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Lightweight CFG analyses.

use crate::{domtree, postorder, Block, Function, Inst, ProgPoint, RegAllocError};
use alloc::vec;
use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

#[derive(Clone, Debug)]
pub struct CFGInfo {
    /// Postorder traversal of blocks.
    pub postorder: Vec<Block>,
    /// Domtree parents, indexed by block.
    pub domtree: Vec<Block>,
    /// For each instruction, the block it belongs to.
    pub insn_block: Vec<Block>,
    /// For each block, the first instruction.
    pub block_entry: Vec<ProgPoint>,
    /// For each block, the last instruction.
    pub block_exit: Vec<ProgPoint>,
    /// For each block, what is the approximate loop depth?
    ///
    /// This measure is fully precise iff the input CFG is reducible
    /// and blocks are in RPO, so that loop backedges are precisely
    /// those whose block target indices are less than their source
    /// indices. Otherwise, it will be approximate, but should still
    /// be usable for heuristic purposes.
    pub approx_loop_depth: Vec<u32>,
    /// For each block, the next block which is at a lower loop depth, or
    /// `Block::invalid` if it is not in a loop.
    pub next_outer_loop: Vec<Block>,
    /// For each block, the previous block which is at a lower loop depth, or
    /// `Block::invalid` if it is not in a loop.
    pub prev_outer_loop: Vec<Block>,
}

impl CFGInfo {
    pub fn new<F: Function>(f: &F) -> Result<CFGInfo, RegAllocError> {
        let postorder = postorder::calculate(f.num_blocks(), f.entry_block(), |block| {
            f.block_succs(block)
        });
        let domtree = domtree::calculate(
            f.num_blocks(),
            |block| f.block_preds(block),
            &postorder[..],
            f.entry_block(),
        );
        let mut insn_block = vec![Block::invalid(); f.num_insts()];
        let mut block_entry = vec![ProgPoint::before(Inst::invalid()); f.num_blocks()];
        let mut block_exit = vec![ProgPoint::before(Inst::invalid()); f.num_blocks()];
        let mut backedge_in = vec![0; f.num_blocks()];
        let mut backedge_out = vec![0; f.num_blocks()];
        let mut next_outer_loop = vec![Block::invalid(); f.num_blocks()];
        let mut prev_outer_loop = vec![Block::invalid(); f.num_blocks()];

        for block in 0..f.num_blocks() {
            let block = Block::new(block);
            for inst in f.block_insns(block).iter() {
                insn_block[inst.index()] = block;
            }
            block_entry[block.index()] = ProgPoint::before(f.block_insns(block).first());
            block_exit[block.index()] = ProgPoint::after(f.block_insns(block).last());

            // Check critical edge condition: if there is more than
            // one predecessor, each must have only one successor
            // (this block).
            let preds = f.block_preds(block).len() + if block == f.entry_block() { 1 } else { 0 };
            if preds > 1 {
                for &pred in f.block_preds(block) {
                    let succs = f.block_succs(pred).len();
                    if succs > 1 {
                        return Err(RegAllocError::CritEdge(pred, block));
                    }
                }
            }

            // Check branch-arg condition: if any successors have more
            // than one predecessor (given above, there will only be
            // one such successor), then the last instruction of this
            // block (the branch) cannot have any args other than the
            // blockparams.
            let mut require_no_branch_args = false;
            for &succ in f.block_succs(block) {
                let preds = f.block_preds(succ).len() + if succ == f.entry_block() { 1 } else { 0 };
                if preds > 1 {
                    require_no_branch_args = true;
                }
            }
            if require_no_branch_args {
                let last = f.block_insns(block).last();
                if !f.inst_operands(last).is_empty() {
                    return Err(RegAllocError::DisallowedBranchArg(last));
                }
            }

            for &succ in f.block_succs(block) {
                if succ.index() <= block.index() {
                    backedge_in[succ.index()] += 1;
                    backedge_out[block.index()] += 1;
                }
            }
        }

        let mut approx_loop_depth = vec![];
        let mut backedge_stack: SmallVec<[usize; 4]> = smallvec![];
        let mut cur_depth = 0;
        let mut max_depth = 0;
        for block in 0..f.num_blocks() {
            if backedge_in[block] > 0 {
                cur_depth += 1;
                max_depth = max_depth.max(cur_depth);
                backedge_stack.push(backedge_in[block]);
            }

            approx_loop_depth.push(cur_depth);

            while backedge_stack.len() > 0 && backedge_out[block] > 0 {
                backedge_out[block] -= 1;
                *backedge_stack.last_mut().unwrap() -= 1;
                if *backedge_stack.last().unwrap() == 0 {
                    cur_depth -= 1;
                    backedge_stack.pop();
                }
            }
        }

        let mut block_at_depth: SmallVec<[Block; 4]> =
            smallvec![Block::invalid(); max_depth as usize];
        for block in 0..f.num_blocks() {
            let cur_depth = approx_loop_depth[block];
            for depth in cur_depth..max_depth {
                block_at_depth[depth as usize] = Block::new(block);
            }
            if cur_depth > 0 {
                prev_outer_loop[block] = block_at_depth[cur_depth as usize - 1];
            }
        }
        block_at_depth.fill(Block::invalid());
        for block in (0..f.num_blocks()).rev() {
            let cur_depth = approx_loop_depth[block];
            for depth in cur_depth..max_depth {
                block_at_depth[depth as usize] = Block::new(block);
            }
            if cur_depth > 0 {
                next_outer_loop[block] = block_at_depth[cur_depth as usize - 1];
            }
        }

        Ok(CFGInfo {
            postorder,
            domtree,
            insn_block,
            block_entry,
            block_exit,
            approx_loop_depth,
            next_outer_loop,
            prev_outer_loop,
        })
    }

    pub fn dominates(&self, a: Block, b: Block) -> bool {
        domtree::dominates(&self.domtree[..], a, b)
    }
}
