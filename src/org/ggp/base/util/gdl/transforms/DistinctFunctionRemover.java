package org.ggp.base.util.gdl.transforms;

import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.util.gdl.grammar.Gdl;
import org.ggp.base.util.gdl.grammar.GdlDistinct;
import org.ggp.base.util.gdl.grammar.GdlFunction;
import org.ggp.base.util.gdl.grammar.GdlLiteral;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.gdl.grammar.GdlRule;
import org.ggp.base.util.gdl.grammar.GdlTerm;

/**
 * This function removes rules with "distinct"s over functions.  For example...
 *
 *    (<= (next (location ?x ?y ?piece))
 *    (does ?player (move ?x1 ?y1 ?x2 ?y2))
 *    (true (location ?x ?y ?piece))
 *    (distinct (f ?x ?y) (f ?x1 ?y1))
 *    (distinct (f ?x ?y) (f ?x2 ?y2)))
 *
 * ...is converted into...
 *
 *    (<= (next (location ?x ?y ?piece))
 *        (does ?player (move ?x1 ?y1 ?x2 ?y2))
 *        (true (location ?x ?y ?piece))
 *        (distinct ?x ?x1)
 *        (distinct ?x ?x2))
 *
 *    (<= (next (location ?x ?y ?piece))
 *        (does ?player (move ?x1 ?y1 ?x2 ?y2))
 *        (true (location ?x ?y ?piece))
 *        (distinct ?x ?x1)
 *        (distinct ?y ?y2))
 *
 *    (<= (next (location ?x ?y ?piece))
 *        (does ?player (move ?x1 ?y1 ?x2 ?y2))
 *        (true (location ?x ?y ?piece))
 *        (distinct ?y ?y1)
 *        (distinct ?x ?x2))
 *
 *    (<= (next (location ?x ?y ?piece))
 *        (does ?player (move ?x1 ?y1 ?x2 ?y2))
 *        (true (location ?x ?y ?piece))
 *        (distinct ?y ?y1)
 *        (distinct ?y ?y2))
 *
 * ...by the expansion of each of the 2 distinct clauses into their 2 constituent parts.
 *
 * This is necessary because the AssignmentIteratorImpl doesn't handle distinct over functions.
 *
 * @author Andrew Rose
 */
public class DistinctFunctionRemover {

    private static final Logger LOGGER = LogManager.getLogger();

    public static List<Gdl> run(List<Gdl> xiDescription) {
        // A single iteration only expands one distinct clause in each rule.  If there's more than one distinct over
        // functions in a single rule, we'll need to run more than once to remove them all.  Therefore, keep running
        // until the number of rules doesn't change.
        int lastSize = 0;
        while (xiDescription.size() != lastSize)
        {
            lastSize = xiDescription.size();
            xiDescription = runOnce(xiDescription);
        }
        return xiDescription;
    }

    private static List<Gdl> runOnce(List<Gdl> xiDescription) {
        List<Gdl> newDescription = new ArrayList<>(xiDescription.size());
        for (Gdl rule : xiDescription) {
            if (!(rule instanceof GdlRule)) {
                // At the top-level, we can have roles, base propositions, init statements and more besides.  Just add
                // these to the new game description.
                newDescription.add(rule);
            } else {
                // This is a rule, so should be processed.
                processRule((GdlRule)rule, newDescription);
            }
        }
        return newDescription;
    }

    private static void processRule(GdlRule xiRule, List<Gdl> xbNewRules) {
        for (GdlLiteral literal : xiRule.getBody()) {
            if (literal instanceof GdlDistinct) {
                GdlDistinct distinct = (GdlDistinct)literal;
                GdlTerm term1 = distinct.getArg1();
                GdlTerm term2 = distinct.getArg2();

                if (!(term1 instanceof GdlFunction && term2 instanceof GdlFunction)) {
                    // This distinct isn't over a pair of functions.  Usually that means it's a pair of variables,
                    // which propnet generation copes with just fine.  However, there could be cases where one's a
                    // function and one isn't.  In this case, the distinct can never be true, so the whole rule can
                    // never be true (because this runs after the DeORer).  So we can just drop it.
                    if (term1 instanceof GdlFunction || term2 instanceof GdlFunction) {
                        LOGGER.warn("Found distinct of function and non-function: " + xiRule);
                        return;
                    }
                    continue;
                }

                GdlFunction fn1 = (GdlFunction)term1;
                GdlFunction fn2 = (GdlFunction)term2;

                if ((!fn1.getName().equals(fn2.getName())) ||
                    (fn1.arity() != fn2.arity())) {
                    // The distincts are either of different functions or the same function with different arities
                    // (which shouldn't happen).  Either way, the literal can never be true, so neither can the rule.
                    LOGGER.warn("Found distinct of different functions / arities: " + xiRule);
                    return;
                }

                // This is exactly the case we should handle.  Rewrite the rule N times, where N is the arity of the
                // function.  In each case, clone the rule but replace the current distinct with a version that looks
                // only at the ith argument to the function.
                LOGGER.debug("Replacing rule: " + xiRule);
                for (int ii = 0; ii < fn1.arity(); ii++) {
                    List<GdlLiteral> newRuleBody = new ArrayList<>(xiRule.getBody().size());
                    for (GdlLiteral literalToCopy : xiRule.getBody()) {
                        if (literalToCopy != literal) {
                            // Just copy across literals that aren't the one we're interested in.
                            newRuleBody.add(literalToCopy);
                        } else {
                            // Replace the distinct in question.
                            newRuleBody.add(GdlPool.getDistinct(fn1.get(ii), fn2.get(ii)));
                        }
                    }

                    // Add the new rule to the output list.
                    GdlRule newRule = GdlPool.getRule(xiRule.getHead(), newRuleBody);
                    xbNewRules.add(newRule);
                    LOGGER.debug("Added new rule: " + newRule);
                }

                // We've replaced the rule that we're working through.  It may still have further distincts that can be
                // treated, but we'll sort that out in the next complete pass.  (If this turns out to be too
                // inefficient, this code can be restructured to just work over the newly created rules.)
                return;
            }
        }

        // This rule didn't have a qualifying distinct literal, so just include the rule verbatim.
        xbNewRules.add(xiRule);
    }
}
