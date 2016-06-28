package org.ggp.base.player.gamer.statemachine.learner;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;

/**
 * A generic position evaluation function, based on neural networks.
 */
public class BreakthroughEvaluationFunction implements EvaluationFunction
{
  private static final Logger LOGGER = LogManager.getLogger();

  private static final int[] sP1Val =
  {
    0, 5,  0, 2,  0, 4,  0, 7,  0,11,  0,16,  0,20,  0,36,
    0,15,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0,15,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0, 5,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0, 5,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0,15,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0,15,  0, 3,  0, 6,  0,10,  0,15,  0,21,  0,28,  0,36,
    0, 5,  0, 2,  0, 4,  0, 7,  0,11,  0,16,  0,20,  0,36,
    0,0
  };

  private static final int[] sP2Val =
  {
    36,0,  20,0,  16,0,  11,0,   7,0,   4,0,  2,0,   5,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,  15,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,  15,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,   5,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,   5,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,  15,0,
    36,0,  28,0,  21,0,  15,0,  10,0,   6,0,  3,0,  15,0,
    36,0,  20,0,  16,0,  11,0,   7,0,   4,0,  2,0,   5,0,
    0,0
  };

  @Override
  public double[] evaluate(ForwardDeadReckonInternalMachineState xiState)
  {
    int lP1Raw = 0;
    int lP2Raw = 0;
    double[] lScores = new double[2];

    int lNumProps = (2 * 8 * 8) + 2;
    OpenBitSet lState = xiState.getContents();
    for (int lii = 0; lii < lNumProps; lii++)
    {
      if (lState.fastGet(lii + xiState.firstBasePropIndex))
      {
        lP1Raw += sP1Val[lii];
        lP2Raw += sP2Val[lii];
      }
    }

    // Score is A / (A + B) which naturally varies from 0 - 1 and is fixed sum.
    double lTotalRaw = lP1Raw + lP2Raw;
    lScores[0] = lP1Raw / lTotalRaw;
    lScores[1] = lP2Raw / lTotalRaw;

    return lScores;
  }

  @Override
  public void sample(ForwardDeadReckonInternalMachineState xiState,
                     double[] xiGoalValues)
  {
    /* Do nothing. */
  }

  @Override
  public void clearSamples()
  {
    /* Do nothing. */
  }

}