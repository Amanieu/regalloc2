//! Fuzz the `regalloc3` register allocator.

use crate::{checker, fuzzing::func};
use arbitrary::{Arbitrary, Result, Unstructured};
use core::cell::RefCell;
use std::thread_local;

/// `regalloc3`-specific options for generating functions.
const OPTIONS: func::Options = func::Options {
    reused_inputs: true,
    fixed_regs: true,
    fixed_nonallocatable: true,
    clobbers: true,
    reftypes: true,
    callsite_ish_constraints: true,
    ..func::Options::DEFAULT
};

/// A convenience wrapper to generate a [`func::Func`] with `regalloc3`-specific
/// options enabled.
#[derive(Clone, Debug)]
pub struct TestCase {
    func: func::Func,
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<TestCase> {
        let func = func::Func::arbitrary_with_options(u, &OPTIONS)?;
        Ok(TestCase { func })
    }
}

/// Test a single function with the `regalloc3` allocator.
///
/// This also:
/// - optionally creates annotations
/// - optionally verifies the incoming SSA
/// - runs the [`checker`].
pub fn check(t: TestCase) {
    let TestCase { func } = &t;
    log::trace!("func:\n{func:?}");

    let env = func::machine_env();
    thread_local! {
        // We test that ctx is cleared properly between runs.
        static CTX: RefCell<crate::Ctx> = RefCell::default();
    }

    CTX.with(|ctx| {
        let ctx = &mut *ctx.borrow_mut();
        ctx.ra3_ctx
            .run(
                func,
                &env,
                &mut ctx.cfginfo,
                &mut ctx.cfginfo_ctx,
                &mut ctx.output,
            )
            .expect("regalloc did not succeed");

        let mut checker = checker::Checker::new(func, &env);
        checker.prepare(&ctx.output);
        checker.run().expect("checker failed");
    });
}

#[test]
fn smoke() {
    arbtest::arbtest(|u| {
        let test_case = TestCase::arbitrary(u)?;
        check(test_case);
        Ok(())
    })
    .budget_ms(1_000);
}
