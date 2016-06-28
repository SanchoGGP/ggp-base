
package org.ggp.base.player.gamer.statemachine.learner;

import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;


public interface EvaluationFunction
{

  /**
   * Evaluate a state according to the evaluation function.
   *
   * @param xiState - the state to evaluate.
   *
   * @return an estimate of the goal values in the specified state.
   */
  public abstract double[] evaluate(ForwardDeadReckonInternalMachineState xiState);

  /**
   * Set a training example for the evaluation function, mapping a state to its value.
   *
   * @param xiState      - the state.
   * @param xiGoalValues - the goal values to learn (in GDL order).
   */
  public abstract void sample(ForwardDeadReckonInternalMachineState xiState,
                              double[] xiGoalValues);

  public abstract void clearSamples();

}