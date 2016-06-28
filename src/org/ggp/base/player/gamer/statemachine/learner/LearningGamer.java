package org.ggp.base.player.gamer.statemachine.learner;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.player.gamer.exception.GamePreviewException;
import org.ggp.base.player.gamer.statemachine.StateMachineGamer;
import org.ggp.base.player.gamer.statemachine.sancho.ThreadControl;
import org.ggp.base.util.game.Game;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.StateMachine;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine;

public class LearningGamer extends StateMachineGamer
{
  private static final Logger LOGGER = LogManager.getLogger();

  private static final boolean BREAKTHROUGH = true;

  private static final int PLY = 5;
  private static final int DUMP_INTERVAL = 10;
  private static final int SAVE_INTERVAL = 100;
  private static final boolean FREEZE = false;

  private static final boolean RELOAD = false;
  //private static final String RELOAD_FROM = "data/games/stanford.breakthroughsmall/evaluation.6hrs.2ply.22500iter.err0.06.nnet";
  private static final String RELOAD_FROM = "data/games/base.breakthrough/evaluation.18hrs.4ply.100iter.nnet";
  private static final boolean TRAIN = true;
  private static final String NAME = "Learner";

  private EvaluationFunction                    mBreakthroughEvalFunc;

  private TrainedEvaluationFunction             mEvalFunc;
  private TrainedEvaluationFunction             mFrozenEvalFunc;

  private ForwardDeadReckonPropnetStateMachine  mUnderlyingStateMachine = null;
  private int                                   mOurRoleIndex;
  private int                                   mTurn = 0;

  @Override
  public StateMachine getInitialStateMachine()
  {
    mUnderlyingStateMachine = new ForwardDeadReckonPropnetStateMachine(ThreadControl.CPU_INTENSIVE_THREADS,
                                                                      getMetaGamingTimeout(),
                                                                      getRole(),
                                                                      mGameCharacteristics);

    System.gc();

    return mUnderlyingStateMachine;
  }

  @Override
  public void stateMachineMetaGame(long xiTimeout)
  {
    mOurRoleIndex = mUnderlyingStateMachine.getRoleOrdering().getOurRawRoleIndex();
    mUnderlyingStateMachine.enableGreedyRollouts(false, true);
    mTurn = 0;

    // Create and initialise a heuristic evaluation function.
    boolean l2PlayerFixedSum = true;
    if (BREAKTHROUGH)
    {
      mBreakthroughEvalFunc = new BreakthroughEvaluationFunction();
    }
    else if (RELOAD)
    {
      LOGGER.info("Reloading saved network");
      mEvalFunc = new TrainedEvaluationFunction(RELOAD_FROM, l2PlayerFixedSum);
      mFrozenEvalFunc = new TrainedEvaluationFunction(RELOAD_FROM, l2PlayerFixedSum);
    }
    else
    {
      mEvalFunc = new TrainedEvaluationFunction(mUnderlyingStateMachine.getBasePropositions().size(),
                                                mUnderlyingStateMachine.getRoles().length,
                                                l2PlayerFixedSum);
      mFrozenEvalFunc = new TrainedEvaluationFunction(mUnderlyingStateMachine.getBasePropositions().size(),
                                                      mUnderlyingStateMachine.getRoles().length,
                                                      l2PlayerFixedSum);
    }

    if (TRAIN && !BREAKTHROUGH)
    {
      // Use TreeStrap to train the evaluation function.
      LOGGER.info("Training");
      treeStrap(xiTimeout);
    }
    else
    {
      // Use the evaluation function to play the game.
      LOGGER.info("Playing");
    }
  }

  private void treeStrap(long xiTimeout)
  {
    double lEpsilon = 0.01;
    int lIterations = 0;
    double lAvgError = 0;
    double lLearningRate = mEvalFunc.INITIAL_LEARNING_RATE;

    while (true /* System.currentTimeMillis() < xiTimeout */)
    {
      // Every so often, update the frozen (target) evaluation function from the training one.  This is necessary for
      // stability.
      if (lIterations % SAVE_INTERVAL == 0)
      {
        if (FREEZE) mFrozenEvalFunc.replaceWith(mEvalFunc);
        mEvalFunc.save();
        lLearningRate = mEvalFunc.cool();
        lEpsilon *= 1.001;
      }

      if (lIterations % DUMP_INTERVAL == 0)
      {
        LOGGER.info("After " + lIterations + " iterations, err = " + (lAvgError * 100.0 / DUMP_INTERVAL) + ", rate = " + lLearningRate + ", epsilon = " + lEpsilon);
        lAvgError = 0;

        if (lIterations % (DUMP_INTERVAL * 10) == 0)
        {
          showSampleGame();
        }
      }

      // Clear the training set.
      mEvalFunc.clearSamples();

      // Do a rollout with n-ply lookahead at each step.
      ForwardDeadReckonInternalMachineState lState =
                                 mUnderlyingStateMachine.createInternalState(mUnderlyingStateMachine.getInitialState());

      while (!mUnderlyingStateMachine.isTerminal(lState))
      {
        // Build a depth-limited game tree, by minimax, using the position evaluation function at the non-terminal
        // leaf nodes.  Builds the training set in the process.
        LearningTree lTree = new LearningTree(mUnderlyingStateMachine, mEvalFunc, FREEZE ? mFrozenEvalFunc : mEvalFunc);
        lTree.search(lState, PLY);

        // Pick the next move epsilon-greedily.
        lState = lTree.epsilonGreedySelection(lState, lEpsilon, false);
      }

      // Train the evaluation function on all the samples gathered during the rollout.
      lAvgError += mEvalFunc.train();

      lIterations++;
    }
  }

  private void showSampleGame()
  {
    LOGGER.info("--- Sample game");

    LearningTree lTree = new LearningTree(mUnderlyingStateMachine, mEvalFunc, FREEZE ? mFrozenEvalFunc : mEvalFunc);
    ForwardDeadReckonInternalMachineState lState =
                                 mUnderlyingStateMachine.createInternalState(mUnderlyingStateMachine.getInitialState());
    while (!mUnderlyingStateMachine.isTerminal(lState))
    {
      // Pick the next move greedily.
      lState = lTree.epsilonGreedySelection(lState, 0, true);
    }
  }

  @Override
  public Move stateMachineSelectMove(long xiTimeout)
  {
    mTurn++;
    LOGGER.info("> > > > > Starting turn " + mTurn + " < < < < <");

    //  Convert to internal rep
    ForwardDeadReckonInternalMachineState currentState = mUnderlyingStateMachine.createInternalState(getCurrentState());

    // Read off the best move according to the learned weights.
    LearningTree lTree;
    if (BREAKTHROUGH)
    {
      lTree = new LearningTree(mUnderlyingStateMachine, mBreakthroughEvalFunc, mBreakthroughEvalFunc);
    }
    else
    {
      lTree = new LearningTree(mUnderlyingStateMachine, mEvalFunc, mEvalFunc);
    }

    Move lBestMove = lTree.bestMove(currentState, mOurRoleIndex, PLY);
    LOGGER.info("Playing: " + lBestMove + " vs immediate best: " + lTree.bestMoveImmediate(currentState, mOurRoleIndex));
    return lBestMove;
  }

  @Override
  public void stateMachineStop()
  {
    // TODO Auto-generated method stub

  }

  @Override
  public void stateMachineAbort()
  {
    // TODO Auto-generated method stub

  }

  @Override
  public void preview(Game xiGame, long xiTimeout) throws GamePreviewException
  {
    // TODO Auto-generated method stub

  }

  @Override
  public String getName()
  {
    return NAME;
  }
}
