package org.ggp.base.player.gamer.statemachine.sancho;

import gnu.trove.map.hash.TObjectLongHashMap;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.player.gamer.statemachine.sancho.MachineSpecificConfiguration.CfgItem;
import org.ggp.base.player.gamer.statemachine.sancho.MoveScoreInfo.MoveScoreInfoAllocator;
import org.ggp.base.player.gamer.statemachine.sancho.RAVEStats.RAVEStatsAllocator;
import org.ggp.base.player.gamer.statemachine.sancho.TreeEdge.TreeEdgeAllocator;
import org.ggp.base.player.gamer.statemachine.sancho.TreeNode.TreeNodeAllocator;
import org.ggp.base.player.gamer.statemachine.sancho.TreePath.TreePathAllocator;
import org.ggp.base.player.gamer.statemachine.sancho.TreePath.TreePathElement;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.Heuristic;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.Heuristic.HeuristicInfo;
import org.ggp.base.player.gamer.statemachine.sancho.pool.CappedPool;
import org.ggp.base.player.gamer.statemachine.sancho.pool.Pool;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveSet;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropositionInfo;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.Factor;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine.MoveWeights;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.StateMachineFilter;
import org.ggp.base.util.statemachine.playoutPolicy.IPlayoutPolicy;
import org.ggp.base.util.statemachine.playoutPolicy.PlayoutPolicyCriticalResponse;

public class MCTSTree
{
  private static final Logger LOGGER = LogManager.getLogger();

  public static final boolean                          FREE_COMPLETED_NODE_CHILDREN                = true;
  public static final boolean                          KEEP_BEST_COMPLETION_PATHS                  = true;
  public static final boolean                          DISABLE_ONE_LEVEL_MINIMAX                   = true;
  private static final boolean                         SUPPORT_TRANSITIONS                         = true;
  public static final int                              MAX_SUPPORTED_BRANCHING_FACTOR              = 3000;
  public static final int                              MAX_SUPPORTED_TREE_DEPTH                    = 500;
  private static final int                             NUM_TOP_MOVE_CANDIDATES                     = 4;

  private final String                                 mTreeDumpFile                               = MachineSpecificConfiguration.getCfgStr(CfgItem.TREE_DUMP);
  private final boolean                                mAllowHyperExpansion;

  // Thresholds governing whether we enable hyper-expansion for a game - the minimums must both be exceeded
  // for it to be enabled
  private static final double                          HYPER_SEQUENCE_MIN_LENGTH_THRESHOLD         = 1.2;
  private static final double                          HYPER_SEQUENCE_MIN_VARIANCE_THRESHOLD       = 0.5;

  /**
   * If goal stability is above a certain threshold we can use interim-state goals to predict final results
   * which makes the use of weight decay and cutoffs appropriate
   */
  private static final double                          GOALS_STABILITY_THRESHOLD                   = 0.8;

  /**
   * The point in the weight decay at which cutoff occurs is set by how many sigmas past the knee
   * (4 is a weight of approximately 0.018, with the knee being the point of symmetry at 0.5)
   */
  private static final double                          CUTOFF_SIGMA                                 = 4;

  /**
   * Whether to use state-similarity measures to heuristically weight move selection
   */
  public static final boolean                          USE_STATE_SIMILARITY_IN_EXPANSION =
                         !MachineSpecificConfiguration.getCfgBool(CfgItem.DISABLE_STATE_SIMILARITY_EXPANSION_WEIGHTING);

  /**
   * Whether to use periodic normalization on node scores
   */
  public final boolean                                 USE_NODE_SCORE_NORMALIZATION;
  /**
   * Whether to use UCB tuned as opposed to simple UCB
   */
  public final boolean                                 USE_UCB_TUNED;

  static final short  MAX_HYPER_RECURSION_DEPTH = 3;

  /**
   * Whether we're creating a database of moves.
   */
  public static final boolean CREATING_DATABASE = MachineSpecificConfiguration.getCfgBool(CfgItem.CREATING_DATABASE);

  /**
   * MixMax bias to use in this game.  0 turns it off entirely
   */
  double mMixiMaxBias = 5;

  private long maxSelectTime = 0;
  private long maxExpandTime = 0;
  public int   maxChildrenSeen = 0;

  volatile TreeNode                                    mRoot = null;
  final boolean                                        mRemoveNonDecisionNodes;
  final boolean                                        mUseEstimatedValueForUnplayedNodes;
  ForwardDeadReckonPropnetStateMachine                 mUnderlyingStateMachine;
  final int                                            mNumRoles;
  final int                                            mWeightDecayKneeDepth;
  final double                                         mWeightDecayScaleFactor;
  final int                                            mWeightDecayCutoffDepth;
  private final int                                    mExpandAfterVisitThreshold;
  final Pool<TreeNode>                                 mNodePool;
  final ScoreVectorPool                                mScoreVectorPool;
  final Pool<TreeEdge>                                 mEdgePool;
  final Pool<TreePath>                                 mPathPool;
  final Pool<RAVEStats>                                mRAVEStatsPool;
  final CappedPool<MoveScoreInfo>                      mCachedMoveScorePool;
  private final TObjectLongHashMap<ForwardDeadReckonInternalMachineState> mPositions;
  int                                                  mSweepInstance                               = 0;
  NodeRefQueue                                         mCompletedNodeRefQueue                       = new NodeRefQueue(512);
  Map<Move, MoveScoreInfo>                             mCousinMoveCache                             = new HashMap<>();
  long                                                 mCousinMovesCachedFor                        = TreeNode.NULL_REF;
  final double[]                                       mRoleRationality;
  final double[]                                       mBonusBuffer;
  final MoveWeights                                    mWeightsToUse;
  final int[]                                          mLatchedScoreRangeBuffer                     = new int[2];
  final int[]                                          mRoleMaxScoresBuffer;
  long                                                 mNumCompletionsProcessed                     = 0;
  Random                                               mRandom                                      = new Random();
  int                                                  mNumTerminalRollouts                         = 0;
  int                                                  mNumNonTerminalRollouts                      = 0;
  int                                                  mNumIncompleteNodes                          = 0;
  int                                                  mNumCompletedBranches                        = 0;
  int                                                  mNumNormalExpansions                         = 0;
  int                                                  mNumAutoExpansions                           = 0;
  int                                                  mMaxAutoExpansionDepth                       = 0;
  int                                                  mNumAllocations                              = 0;
  int                                                  mNumTranspositions                           = 0;
  double                                               mAverageAutoExpansionDepth                   = 0;
  boolean                                              mCompleteSelectionFromIncompleteParentWarned = false;
  int                                                  mNumReExpansions                             = 0;
  Heuristic                                            mHeuristic;
  final RoleOrdering                                   mRoleOrdering;
  final Role                                           mOurRole;
  RolloutProcessorPool                                 mRolloutPool;
  RuntimeGameCharacteristics                           mGameCharacteristics;
  final Factor                                         mFactor;
  final StateMachineFilter                             mSearchFilter;
  boolean                                              mEvaluateTerminalOnNodeCreation;
  private final TreeNodeAllocator                      mTreeNodeAllocator;
  final TreeEdgeAllocator                              mTreeEdgeAllocator;
  final MoveScoreInfoAllocator                         mMoveScoreInfoAllocator;
  private final TreePathAllocator                      mTreePathAllocator;
  final RAVEStatsAllocator                             mRAVEStatsAllocator;
  final GameSearcher                                   mGameSearcher;
  final StateSimilarityMap                             mStateSimilarityMap;
  private final ForwardDeadReckonInternalMachineState  mNonFactorInitialState;
  public boolean                                       mIsIrrelevantFactor = false;
  private final DataOutputStream                       mDatabase;

  /**
   * The highest score seen in the current turn (for our role).
   */
  public int                              mHighestRolloutScoreSeen;

  /**
   * The lowest score seen in the current turn (for our role).
   */
  public int                              mLowestRolloutScoreSeen;

  /**
   * Shallowest observed completion depth for the game - updated dynamically
   */
  int                                                 mShallowestCompletionDepth;

  // Scratch variables for tree nodes to use to avoid unnecessary object allocation.
  // Note - several of these could probably be collapsed into a lesser number since they are not
  // concurrently used, but it's not worth the risk currently
  final HeuristicInfo                                 mNodeHeuristicInfo;
  final double[]                                      mNodeAverageScores;
  final double[]                                      mNodeAverageSquaredScores;
  final RolloutRequest                                mNodeSynchronousRequest;
  final ForwardDeadReckonLegalMoveInfo[]              mNodeTopMoveCandidates;
  final ForwardDeadReckonInternalMachineState         mStateScratchBuffer;
  final ForwardDeadReckonInternalMachineState[]       mChildStatesBuffer;
  final ForwardDeadReckonPropositionInfo[]            mRoleControlProps;
  final ForwardDeadReckonInternalMachineState         mNextStateBuffer;
  final ForwardDeadReckonLegalMoveInfo[]              mJointMoveBuffer;
  final ForwardDeadReckonLegalMoveInfo[]              mFastForwardPartialMoveBuffer;
  final double[]                                      mCorrectedAverageScoresBuffer;
  final double[]                                      mBlendedCompletionScoreBuffer;

  public MCTSTree(ForwardDeadReckonPropnetStateMachine xiStateMachine,
                  Factor xiFactor,
                  Pool<TreeNode> xiNodePool,
                  ScoreVectorPool xiScorePool,
                  Pool<TreeEdge> xiEdgePool,
                  Pool<TreePath> xiPathPool,
                  Pool<RAVEStats> xiRAVEStatsPool,
                  RoleOrdering xiRoleOrdering,
                  RolloutProcessorPool xiRolloutPool,
                  RuntimeGameCharacteristics xiGameCharacteristics,
                  Heuristic xiHeuristic,
                  GameSearcher xiGameSearcher,
                  ForwardDeadReckonPropositionInfo[] roleControlProps)
  {
    mUnderlyingStateMachine = xiStateMachine;
    mNumRoles = xiStateMachine.getRoles().length;
    mStateSimilarityMap = (MachineSpecificConfiguration.getCfgBool(CfgItem.DISABLE_STATE_SIMILARITY_EXPANSION_WEIGHTING) ? null : new StateSimilarityMap(xiStateMachine.getFullPropNet(), xiNodePool));
    mRoleControlProps = roleControlProps;
    mNodePool = xiNodePool;
    mScoreVectorPool = xiScorePool;
    mEdgePool = xiEdgePool;
    mPathPool = xiPathPool;
    mRAVEStatsPool = xiRAVEStatsPool;
    mFactor = xiFactor;
    if (mFactor != null)
    {
      mSearchFilter = mFactor;
    }
    else
    {
      mSearchFilter = xiStateMachine.getBaseFilter();
    }

    //  Initialize shallowest completion depth to estimate from metagaming
    mShallowestCompletionDepth = xiGameCharacteristics.getEarliestCompletionDepth();

    //  Hyper-expansion is enabled if allowed by the config and the game exhibits sufficiently common hyper-sequences with
    //  a sufficiently variable length to make it worthwhile
    mAllowHyperExpansion = MachineSpecificConfiguration.getCfgBool(CfgItem.ALLOW_HYPEREXPANSION) &&
                           xiGameCharacteristics.getAverageHyperSequenceLength() > HYPER_SEQUENCE_MIN_LENGTH_THRESHOLD &&
                           xiGameCharacteristics.getVarianceHyperSequenceLength() > HYPER_SEQUENCE_MIN_VARIANCE_THRESHOLD;
    if ( mAllowHyperExpansion)
    {
      LOGGER.info("Hyper expansion processing is enabled");
    }
    else
    {
      LOGGER.info("Hyper expansion processing is not enabled");
    }

    //  Most of the time we don't want to create tree nodes representing states in which
    //  only a single choice is available to whichever role is choosing at that node.
    //  However we continue to do so in two cases:
    //    1) Single player games - this is necessary to preserve plan-generation by (simple)
    //       backward walking of the tree from a win-terminal node
    //    2) Simultaneous move games, because we must assess the impact of a move in a cousin
    //       to cope with partial information, and a forced choice in one node does not imply
    //       a forced choice in all cousins
    mRemoveNonDecisionNodes = (mNumRoles > 1 && xiGameCharacteristics.hasAdequateSampling && !xiGameCharacteristics.isSimultaneousMove && (!xiGameCharacteristics.isPseudoSimultaneousMove || xiFactor != null));

    //  Apply decay and cutoff if either:
    //    1)  The goals are sufficiently stable (goals in a non-terminal state are a good predictor
    //        of final result)
    //    2)  All of the following apply:
    //        2.1) Non-draws cluster at shallower depths on average than do draws
    //        2.2) Mean game length is not much past median game length (which would imply results tend to happen late)
    //        2.3) Max-length games are almost all draws
    //        These conditions are a proxy for a more general (but harder) analysis of the distribution of results
    //        wherein win vs draw distribution over depth tends not to increase.  Mostly it is intended to capture
    //        the use of decay in games that have artificial draw-after-N-turns terminal conditions and where the non-draws
    //        can happen a lot earlier
    if (xiGameCharacteristics.numRoles > 1 &&
        xiGameCharacteristics.getAverageNonDrawLength() > 0 &&
        (xiGameCharacteristics.getGoalsStability() > GOALS_STABILITY_THRESHOLD ||
         (xiGameCharacteristics.getLongDrawsProportion() > 0.8 &&
          xiGameCharacteristics.getAverageNonDrawLength() <= xiGameCharacteristics.getAverageLength() &&
          xiGameCharacteristics.getAverageLength() < (xiGameCharacteristics.getMaxLength()+xiGameCharacteristics.getMinLength())*1.05/2.0)))
    {
      //  If goals are stable the decay depth is somewhat arbitrary, but we want finishes to be plausibly 'in range'
      //  so we use the shortest length seen from the initial state.
      //  If goals are NOT stable and we are using decay based on seeing
      //  non-draw results earlier than average finishes we use that non-draw average length
      if (xiGameCharacteristics.getGoalsStability() > GOALS_STABILITY_THRESHOLD)
      {
        mWeightDecayKneeDepth = xiGameCharacteristics.getMinLength();
      }
      else
      {
        mWeightDecayKneeDepth = (int)xiGameCharacteristics.getAverageNonDrawLength();
      }
      //  Steepness of the cutoff is proportional to the depth of the knee (so basically we use
      //  a scale-free shape for decay) - this is an empirical decision and seems to be better than using
      //  std deviation of game length
      mWeightDecayScaleFactor = (double)mWeightDecayKneeDepth/6;
      //  Cutoff set to occur at fixed decay factor
      mWeightDecayCutoffDepth = mWeightDecayKneeDepth + (int)(mWeightDecayScaleFactor*CUTOFF_SIGMA);
      LOGGER.info("Weight decay knee and scale factor: (" + mWeightDecayKneeDepth + ", " + mWeightDecayScaleFactor + ")");
    }
    else
    {

      mWeightDecayKneeDepth = -1;
      mWeightDecayScaleFactor = 0;
      mWeightDecayCutoffDepth = 1000;

      LOGGER.info("Weight decay disabled");
    }

    if ( xiGameCharacteristics.getAverageBranchingFactor() > 20 && xiGameCharacteristics.getTerminalityDensity() > 0 )
    {
      //  Empirical formula set from Majorities and Hex and currently only used for high branching factor games
      //  Limit how extreme this can get in case of exceptional games outside current tuning (to a max of 8 for now)
      //  Motivation is that inn high BF games expansion is expansive, so we want to not bother on nodes that
      //  never get much played through anyway.  Conversely if it would have turned out that a child of such an
      //  expansion were terminal we would gain a lot of rapid pruning, so we want to do this less in games with a high
      //  density of terminal states throughout the state space.
      //  Prior to IGGP2015 only really tuned for Hex and Majorities, so disabled entirely for anything with low-medium
      //  branching factor
      mExpandAfterVisitThreshold = Math.min(8, (int)(xiGameCharacteristics.getAverageBranchingFactor()*0.16 + 0.006/xiGameCharacteristics.getTerminalityDensity()));
    }
    else
    {
      mExpandAfterVisitThreshold = 0;
    }

    LOGGER.info("Expand-after-visits threshold set to " + mExpandAfterVisitThreshold);

    //  We do not normalize puzzles as this seems to be detrimental.  Normalization is helpful when initial convergence
    //  is misleading (i.e. - non-monotonic), which is probably much rarer in puzzles (where there is no agent acting
    //  against us).  However, this remains just a theory.
    //  Empirical results also indicate normalization does not work well in games with either old-style
    //  heuristics or which use hyper-expansion
    USE_NODE_SCORE_NORMALIZATION = !xiGameCharacteristics.isPseudoPuzzle &&
                                   MachineSpecificConfiguration.getCfgBool(CfgItem.USE_NODE_SCORE_NORMALIZATION) &&
                                   !xiHeuristic.applyAsSimpleHeuristic() &&
                                   !mAllowHyperExpansion;
    if (USE_NODE_SCORE_NORMALIZATION)
    {
      LOGGER.info("Using periodic node score normalization");
    }
    else
    {
      LOGGER.info("Not using periodic node score normalization");
    }

    USE_UCB_TUNED = MachineSpecificConfiguration.getCfgBool(CfgItem.USE_UCB_TUNED);
    if (USE_UCB_TUNED)
    {
      LOGGER.info("Using UCB-tuned");
    }
    else
    {
      LOGGER.info("Using simple UCB");
    }

    if (mWeightDecayCutoffDepth >= 1000)
    {
      LOGGER.info("Early cutoff disabled");
    }
    else
    {
      LOGGER.info("Early cutoff depth: " + mWeightDecayCutoffDepth);
    }

    //  Currently enable use of MixMax selection only in very specific cases.  This is because there
    //  has not been sufficient time to set up tuning parameters, so we are just going for a couple of cases
    //  we know it works well in
    if ( xiGameCharacteristics.getIsFixedSum() && mNumRoles == 2 &&
         ((mWeightDecayCutoffDepth < 1000 && !xiGameSearcher.mUseRAVE && xiGameCharacteristics.getAverageBranchingFactor() < 10) ||  // TTCC4-like
          (xiGameSearcher.mUseRAVE && xiGameCharacteristics.getAverageBranchingFactor() < 10 && xiGameCharacteristics.getChoicesHighWaterMark(0) > 50)) ) // NBTTT-like
    {
      mMixiMaxBias = 5;

      LOGGER.info("Using Mixi-max bias of " + mMixiMaxBias);
    }
    else
    {
      mMixiMaxBias = 0;

      LOGGER.info("Mixi-max selection disabled");
    }

    mRoleOrdering = xiRoleOrdering;
    mOurRole = xiRoleOrdering.roleIndexToRole(0);
    mHeuristic = xiHeuristic;
    mGameCharacteristics = xiGameCharacteristics;
    mRolloutPool = xiRolloutPool;
    mPositions = new TObjectLongHashMap<>((int)(mNodePool.getCapacity() / 0.75f), 0.75f, TreeNode.NULL_REF);

    //  For now we only automatically enable use of estimated values for unplayed nodes (in select)
    //  in games with negative goal latches, which amounts to ELB.  Further testing is needed, so for
    //  now wider enablement requires an explicit config setting.
    mUseEstimatedValueForUnplayedNodes = MachineSpecificConfiguration.getCfgBool(CfgItem.ENABLE_INITIAL_NODE_ESTIMATION) |
                                        mUnderlyingStateMachine.mLatches.hasNegativelyLatchedGoals();

    if (mUseEstimatedValueForUnplayedNodes)
    {
      LOGGER.info("Estimated initial values for nodes with no play-throughs is enabled");
    }
    if (xiFactor != null)
    {
      mNonFactorInitialState = xiStateMachine.createInternalState(xiStateMachine.getInitialState());
      mNonFactorInitialState.intersect(xiFactor.getInverseStateMask(false));
    }
    else
    {
      mNonFactorInitialState = null;
    }

    mEvaluateTerminalOnNodeCreation = !mGameCharacteristics.getIsFixedMoveCount() || xiGameSearcher.mUseGoalGreedy;

    mNumCompletionsProcessed = 0;
    mCompleteSelectionFromIncompleteParentWarned = false;
    mTreeNodeAllocator = new TreeNodeAllocator(this);
    mTreeEdgeAllocator = new TreeEdgeAllocator();
    mTreePathAllocator = new TreePathAllocator(this);
    mRAVEStatsAllocator = new RAVEStatsAllocator(mGameCharacteristics);
    mGameSearcher = xiGameSearcher;

    mBonusBuffer = new double[mNumRoles];
    mRoleRationality = new double[mNumRoles];
    mRoleMaxScoresBuffer = new int[mNumRoles];
    mWeightsToUse = xiStateMachine.createMoveWeights();

    //  For now assume players in muli-player games are somewhat irrational.
    //  FUTURE - adjust during the game based on correlations with expected
    //  scores
    for (int i = 0; i < mNumRoles; i++)
    {
      if (xiGameCharacteristics.numRoles > 2)
      {
        mRoleRationality[i] = (i == 0 ? 1 : 0.8);
      }
      else
      {
        mRoleRationality[i] = 1;
      }
    }

    // If we're creating a database, open it now.
    DataOutputStream lDatabase = null;
    if (CREATING_DATABASE)
    {
      LOGGER.info("Creating state database");
      try
      {
        lDatabase = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("states.db")));
      }
      catch (IOException lEx)
      {
        LOGGER.warn("Failed to create state database", lEx);
      }
    }
    mDatabase = lDatabase;

    // Create the variables used by TreeNodes to avoid unnecessary object allocation.
    mNodeHeuristicInfo            = new HeuristicInfo(mNumRoles);
    mNodeAverageScores            = new double[mNumRoles];
    mNodeAverageSquaredScores     = new double[mNumRoles];
    mNodeSynchronousRequest       = new RolloutRequest(mNumRoles, mUnderlyingStateMachine);
    mNodeTopMoveCandidates        = new ForwardDeadReckonLegalMoveInfo[NUM_TOP_MOVE_CANDIDATES];
    mCorrectedAverageScoresBuffer = new double[mNumRoles];
    mJointMoveBuffer              = new ForwardDeadReckonLegalMoveInfo[mNumRoles];
    mFastForwardPartialMoveBuffer = new ForwardDeadReckonLegalMoveInfo[mNumRoles];
    mBlendedCompletionScoreBuffer = new double[mNumRoles];
    mNextStateBuffer              = mUnderlyingStateMachine.createEmptyInternalState();
    mChildStatesBuffer            = new ForwardDeadReckonInternalMachineState[MAX_SUPPORTED_BRANCHING_FACTOR];
    for (int lii = 0; lii < MAX_SUPPORTED_BRANCHING_FACTOR; lii++)
    {
      mChildStatesBuffer[lii] = mUnderlyingStateMachine.createEmptyInternalState();
    }
    mStateScratchBuffer = mUnderlyingStateMachine.createEmptyInternalState();
    mMoveScoreInfoAllocator = new MoveScoreInfoAllocator(mNumRoles);
    mCachedMoveScorePool = new CappedPool<>(MAX_SUPPORTED_BRANCHING_FACTOR);
  }

  public void empty()
  {
    mNumCompletedBranches = 0;
    mNumNormalExpansions = 0;
    mNumAutoExpansions = 0;
    mMaxAutoExpansionDepth = 0;
    mAverageAutoExpansionDepth = 0;
    mRoot = null;
    mNodePool.clear(mTreeNodeAllocator, true);
    mPositions.clear();
    mNumIncompleteNodes = 0;
  }

  /**
   * Terminate the MCTS tree.  It must not be used again after this call.
   */
  public void terminate()
  {
    empty();
    if (mDatabase != null)
    {
      IOUtils.closeQuietly(mDatabase);
    }
  }
  TreeNode allocateNode(ForwardDeadReckonInternalMachineState state,
                        TreeNode parent,
                        boolean disallowTransposition)
  {
    TreeNode result = ((state != null && !disallowTransposition) ? findTransposition(state) : null);

    //validateAll();
    //  Use of pseudo-noops in factors can result in recreation of the root state (only)
    //  a lower level with a joint move of (pseudo-noop, noop, noop, ..., noop).  This
    //  must not be linked back to or else a loop will be created
    if ((!SUPPORT_TRANSITIONS || result == null) || result == mRoot)
    {
      //LOGGER.debug("Add state " + state);
      result = mNodePool.allocate(mTreeNodeAllocator);

      //if (positions.values().contains(result))
      //{
      //  LOGGER.info("Node already referenced by a state!");
      //}
      if (state != null)
      {
        result.setState(state);
        if (!disallowTransposition)
        {
          addToTranspositionIndexes(result);
        }
      }
      assert(!result.mFreed) : "Bad ref in positions table";
    }
    else
    {
      assert(!result.mFreed) : "Bad ref in positions table";

      mNumTranspositions++;
    }

    if (parent != null)
    {
      result.addParent(parent);

      //parent.adjustDescendantCounts(result.descendantCount+1);
    }

    mNumAllocations++;

    //validateAll();
    return result;
  }

  /**
   * @return total number of logical node allocations made
   */
  public int getNumAllocations()
  {
    return mNumAllocations;
  }

  /**
   * @return total number of nodes allocations made that
   * were transpositions
   */
  public int getNumTranspositions()
  {
    return mNumTranspositions;
  }

  /**
   * Inform the tree that a node is being freed.
   *
   * @param xiTreeNode - the node that is being freed.
   */
  public void nodeFreed(TreeNode xiTreeNode)
  {
    removeFromTranspositionIndexes(xiTreeNode);
  }

  public void addToTranspositionIndexes(TreeNode xiTreeNode)
  {
    if (SUPPORT_TRANSITIONS)
    {
      assert(!mPositions.containsKey(xiTreeNode.mState));
      mPositions.put(xiTreeNode.mState, xiTreeNode.getRef());
    }
  }

  public void removeFromTranspositionIndexes(TreeNode xiTreeNode)
  {
    if (SUPPORT_TRANSITIONS)
    {
      long lRef = mPositions.get(xiTreeNode.mState);

      if (lRef == TreeNode.NULL_REF)
      {
        return;
      }
      else if (lRef == xiTreeNode.getRef())
      {
        mPositions.remove(xiTreeNode.mState);
      }
      else
      {
        //assert(false) : "Attempt to remove incorrect node from position indexes";
      }
    }
  }

  public TreeNode findTransposition(ForwardDeadReckonInternalMachineState xiState)
  {
    if (SUPPORT_TRANSITIONS)
    {
      long lRef = mPositions.get(xiState);

      if (lRef == TreeNode.NULL_REF)
      {
        return null;
      }

      TreeNode lNode = TreeNode.get(mNodePool, lRef);
      assert(lNode != null);
      assert(xiState.equals(lNode.mState));
      assert(!mRemoveNonDecisionNodes || lNode == mRoot || lNode.mComplete || lNode.mNumChildren != 1);

      return lNode;
    }

    return null;
  }

  void processNodeCompletions()
  {
    while (!mCompletedNodeRefQueue.isEmpty())
    {
      //validateAll();
      TreeNode lNode = TreeNode.get(mNodePool, mCompletedNodeRefQueue.remove());
      if (lNode != null)
      {
        assert(!lNode.mFreed) : "Freed node returned from pool";
        //  New root will always be marked as incomplete on setting a new root state so
        //  it can be re-expanded.  If it happens to be on the completion queue at the time
        //  that can lead to this effect (which is harmless)
        assert(lNode.mComplete || lNode == lNode.mTree.mRoot) : "Queued node was not complete";
        lNode.processCompletion();
      }
    }
  }

  void makeFactorState(ForwardDeadReckonInternalMachineState state)
  {
    state.intersect(mFactor.getStateMask(false));
    //  Set the rest of the state to 'neutral' values.  We use the initial state
    //  as this is guaranteed to be legal and non-terminal
    state.merge(mNonFactorInitialState);
  }

  public void setRootState(ForwardDeadReckonInternalMachineState state, short rootDepth, ForwardDeadReckonLegalMoveInfo lastMove)
  {
    ForwardDeadReckonInternalMachineState factorState;

    if (mFactor == null)
    {
      factorState = state;
    }
    else
    {
      factorState = new ForwardDeadReckonInternalMachineState(state);
      makeFactorState(factorState);
    }

    if (mRoot == null)
    {
      //oldRoot = null;
      mRoot = allocateNode(factorState, null, true);
      mRoot.mDecidingRoleIndex = mNumRoles - 1;
      mRoot.setDepth(rootDepth);
    }
    else
    {
      TreeNode previousChoiceNode = mRoot;

      while(previousChoiceNode != null && previousChoiceNode.mNumChildren == 1)
      {
        Object choice = previousChoiceNode.mChildren[0];
        if ( choice instanceof TreeEdge)
        {
          previousChoiceNode = mRoot.get(((TreeEdge)choice).getChildRef());
        }
        else
        {
          break;
        }
      }

      if (mRoot.mState.equals(factorState))
      {
        assert (rootDepth == 0 || mFactor != null);
        if (rootDepth == 0)
        {
          calculateCriticalResponses();

          //  This is the start of the first turn, after some searching at the end of meta-gaming
          //  If the root score variance is 0 and this is a factored game, we mark this factor as
          //  uninteresting, and will henceforth spend no time searching it
          if (mFactor != null &&
              mRoot.mNumVisits > 500 &&
              mLowestRolloutScoreSeen == mHighestRolloutScoreSeen)
          {
            mIsIrrelevantFactor = true;

            LOGGER.info("Identified irrelevant factor - supressing search");
          }
        }
      }
      else
      {
        calculateCriticalResponses();

        TreeNode existingRootStateNode = findTransposition(factorState);

        //  In a factorized game there will be pseudo-noops, and the nodes generated by traversing the
        //  pseudo-noop from the root is not added to the transposition tables, and hence can result in the
        //  state we are searching for not being found, so we need to check that also
        if (existingRootStateNode == null && mFactor != null)
        {
          existingRootStateNode = mRoot.findNode(factorState, mUnderlyingStateMachine.getRoles().length + 1);
        }

        assert(existingRootStateNode != null || mRoot.findNode(factorState, mUnderlyingStateMachine.getRoles().length + 1)==null);
        //assert(existingRootStateNode == null || existingRootStateNode.getDepth() > root.getDepth());

        //  In games with forced moves links will go from states-with-choices to other states-with-choices, so
        //  intermediate nodes (with no choices) will not exists.  This means that as the actual game path traverses
        //  such forced steps, and passes through such an intermediary state, it will not be found.  To hook up with
        //  the existing tree we need to 'fast-forward' from this node to the next (forced) state in order to find
        //  the point of linkage into the existing tree
        //  In the case where the state IS found, either it's our turn, in which case the existing choice node for this state
        //  can be used directly as the new root, or (and this can only happen if we are
        //  collapsing non-choice nodes) the existing node indexed for the root state is
        //  another role's choice.  In the latter case we need a 'proxy' node for the root
        //  so that it has the correct (singular) choice ready for getBestMove().  That
        //  singular choice will then point to the extant root state node, which is the
        //  actual decision node.
        TreeNode oldRoot = mRoot;

        if (existingRootStateNode != null && existingRootStateNode.mDecidingRoleIndex == mNumRoles - 1)
        {
          assert(oldRoot.linkageValid());

          assert(existingRootStateNode.linkageValid());
          mRoot = existingRootStateNode;
          assert(mRoot.mParents.size()>0);
        }
        else
        {
          assert(existingRootStateNode == null || mRemoveNonDecisionNodes);
          assert(existingRootStateNode == null || existingRootStateNode.linkageValid());

          //  Allocate proxy node for the root
          mRoot = allocateNode(factorState, null, true);
          mRoot.mDecidingRoleIndex = mNumRoles - 1;
          mRoot.setDepth(rootDepth);

          //  Expand to get the correct (singular) choice
          mRoot.expand(null, mJointMoveBuffer, rootDepth-1);

          assert(existingRootStateNode == null || mRoot.mNumChildren == 1 || oldRoot.mComplete);

          //  In the forced-move case check to see if we can link into the existing search tree
          if (mRoot.mNumChildren == 1)
          {
            TreeEdge selected;

            if (mRoot.mChildren[0] instanceof ForwardDeadReckonLegalMoveInfo)
            {
              //  Create an edge for the singular choice
              selected = mEdgePool.allocate(mTreeEdgeAllocator);
              selected.setParent(mRoot, (ForwardDeadReckonLegalMoveInfo)mRoot.mChildren[0]);
              mRoot.mChildren[0] = selected;
            }
            else
            {
              selected = (TreeEdge)mRoot.mChildren[0];
            }

            mJointMoveBuffer[0] = selected.mPartialMove;

            if (existingRootStateNode == null)
            {
              if (selected.getChildRef() == TreeNode.NULL_REF)
              {
                mRoot.createChildNodeForEdge(selected, mJointMoveBuffer);
              }

              TreeNode createdForcedMoveChild = mRoot.get(selected.getChildRef());
              if (!createdForcedMoveChild.mComplete && createdForcedMoveChild.isUnexpanded())
              {
                TreePath visited = mPathPool.allocate(mTreePathAllocator);

                visited.push(mRoot, selected);
                createdForcedMoveChild = createdForcedMoveChild.expand(visited, mJointMoveBuffer, rootDepth);
              }

              //  Have now created the first decision node and joined it into the existing tree
              //  if it existed there.
              existingRootStateNode = createdForcedMoveChild;
            }
            else
            {
              if (selected.getChildRef() == TreeNode.NULL_REF)
              {
                selected.setChild(existingRootStateNode);
                existingRootStateNode.addParent(mRoot);
              }
              else
              {
                assert(existingRootStateNode == mRoot.get(selected.getChildRef()));
              }
            }

            assert(mRoot.linkageValid());

            //  Set the root's count stats to those of the extant choice node it is effectively
            //  proxying
            selected.setNumVisits(existingRootStateNode.mNumVisits);
            for(int i = 0; i < mNumRoles; i++)
            {
              mRoot.setAverageScore(i, existingRootStateNode.getAverageScore(i));
              mRoot.setAverageSquaredScore(i, existingRootStateNode.getAverageSquaredScore(i));
            }
            mRoot.mNumVisits = existingRootStateNode.mNumVisits;
            mRoot.mNumUpdates = existingRootStateNode.mNumUpdates;

            if (existingRootStateNode.mComplete)
            {
              //  There are two possible reasons the old root could have been marked as
              //  complete - either it was complete because of the state of its children
              //  in which case we want to re-propagate that, or else it had been marked
              //  complete by local search without known path (in which case we must re-find
              //  the completion and so clear the completion)
              if (existingRootStateNode.mTerminal)
              {
                mRoot.markComplete(existingRootStateNode, (short)existingRootStateNode.getDepth());
              }
              else
              {
                existingRootStateNode.mComplete = false;
                if (!existingRootStateNode.isUnexpanded())
                {
                  existingRootStateNode.checkChildCompletion(false);
                }
              }
            }
          }
        }

        int percentageFreed = oldRoot.freeAllBut(mRoot);

        if ( previousChoiceNode != null && previousChoiceNode.mNumChildren > 1 && mGameSearcher.getPlan().isEmpty() )
        {
          //  If there was an unexpectedly large amount freed that indicates an unexpected opponent
          //  move
          int maxExpectedFreed = (100*previousChoiceNode.mNumChildren - 100)/previousChoiceNode.mNumChildren;
          if ((percentageFreed > maxExpectedFreed) && (!mRoot.mComplete))
          {
            String unexpectedMoveMessage;

            if ( mRandom.nextInt(2) == 0 )
            {
              unexpectedMoveMessage = "Hmmm...I wasn't expecting that" + (lastMove == null ? "" : " " + lastMove) + ", did I miss something?";
            }
            else
            {
              unexpectedMoveMessage = "That was not something I expected" + (lastMove == null ? "" : " " + lastMove) + "!  Are you sure you really wanted to do that?";
            }
            if ( mGameSearcher.getBroadcaster() != null )
            {
              mGameSearcher.getBroadcaster().broadcast(unexpectedMoveMessage);
            }
          }
        }
        assert(existingRootStateNode == null || existingRootStateNode == mRoot || existingRootStateNode.mParents.size()==1);
        assert(mRoot.mParents.size() == 0);
      }
    }

    assert(!mRoot.mFreed) : "Root node has been freed";
    assert(mRoot.mParents.size() == 0);
    //validateAll();

    //  Special case - because we can mark nodes complete before they are terminal if greedy rollouts are
    //  in use, it is possible for a complete root to not have a known child path as the root hits
    //  the penultimate move.  In such circumstances we must force re-expansion to identify the correct last
    //  move
    if (mRoot.mComplete && mUnderlyingStateMachine.getIsGreedyRollouts())
    {
      boolean foundChildPath = false;

      for(int i = 0; i < mRoot.mNumChildren; i++ )
      {
        if (mRoot.mChildren[i] instanceof TreeEdge &&
            ((TreeEdge)mRoot.mChildren[i]).getChildRef() != TreeNode.NULL_REF)
        {
          TreeNode child = mRoot.get(((TreeEdge)mRoot.mChildren[i]).getChildRef());
          if (child != null && child.mComplete && child.getAverageScore(0) == mRoot.getAverageScore(0))
          {
            foundChildPath = true;
            break;
          }
        }
      }

      if (!foundChildPath)
      {
        LOGGER.info("Complete root has no known child path - marking incomplete to force re-expansion");
        mRoot.mComplete = false;
      }
    }
    if (mRoot.mNumChildren == 0)
    {
      LOGGER.info("Encountered childless root - must re-expand");

      if (mRoot.mComplete)
      {
        mNumCompletedBranches--;
      }
      mRoot.mComplete = false;
      //  Latched score detection can cause a node that is not strictly terminal (but has totally
      //  fixed scores for all subtrees) to be flagged as terminal - we must reset this to ensure
      //  it get re-expanded one level (from which we'll essentially make a random choice)
      mRoot.mTerminal = false;

      //  Must expand here as async activity on the local search can mark the root complete again
      //  before an expansion takes place if we let control flow out of the synchronized section
      //  before expanding
      mRoot.expand(null, mJointMoveBuffer, rootDepth-1);
    }

    LOGGER.info("Root has " + mRoot.mNumChildren + " children, and is " + (mRoot.mComplete ? "complete" : "not complete"));

    mLowestRolloutScoreSeen = 1000;
    mHighestRolloutScoreSeen = -100;

    mHeuristic.newTurn(mRoot.mState, mRoot);
  }

  /**
   * Perform a single MCTS expansion.
   *
   * @param forceSynchronous
   * @param xiChosenMove - the move which has already been chosen (for us) this turn.
   *
   * @return whether the tree is now fully explored.
   *
   * @throws MoveDefinitionException
   * @throws TransitionDefinitionException
   * @throws GoalDefinitionException
   * @throws InterruptedException
   */
  public boolean growTree(boolean forceSynchronous, Move xiChosenMove)
    throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException
  {
    //  In an irrelevant factor we don't want to waste time searching - just need
    //  to do enough for the children to be enumerable
    if (mIsIrrelevantFactor && !mRoot.hasUnexpandedChoices())
    {
      return false;
    }

    //validateAll();
    //validationCount++;
    selectAction(forceSynchronous, xiChosenMove);
    processNodeCompletions();
    return mRoot.mComplete;
  }

  /**
   * @param firstDecision if false choice from the root always else choice from first node that has one
   * @return the best move from this tree.
   */
  public FactorMoveChoiceInfo getBestMove(boolean firstDecision)
  {
    FactorMoveChoiceInfo bestMoveInfo = mRoot.getBestMove(true, null, firstDecision);

    if (!firstDecision)
    {
      bestMoveInfo.logBest();
      LOGGER.info("Num nodes in use: " + mNodePool.getNumItemsInUse());
      LOGGER.info("Num true rollouts added: " + mNumNonTerminalRollouts);
      LOGGER.info("Num terminal nodes revisited: " + mNumTerminalRollouts);
      LOGGER.info("Num incomplete nodes: " + mNumIncompleteNodes);
      LOGGER.info("Num completely explored branches: " + mNumCompletedBranches);
      if (mNumAutoExpansions + mNumNormalExpansions > 0)
      {
        LOGGER.info("Percentage forced single-choice expansion: " +
                    ((double)mNumAutoExpansions / (mNumAutoExpansions + mNumNormalExpansions)));
        LOGGER.info("Average depth of auto-expansion instances: " + mAverageAutoExpansionDepth);
        LOGGER.info("Maximum depth of auto-expansion instances: " + mMaxAutoExpansionDepth);
      }
      LOGGER.info("Current observed rollout score range: [" +
                  mLowestRolloutScoreSeen + ", " +
                  mHighestRolloutScoreSeen + "]");

      mNumNonTerminalRollouts = 0;
      mNumTerminalRollouts = 0;
    }

    if ( mTreeDumpFile != null )
    {
      String extension;
      String stem;
      int lastDotIndex = mTreeDumpFile.lastIndexOf('.');
      if ( lastDotIndex != -1 )
      {
        extension = mTreeDumpFile.substring(lastDotIndex);
        stem = mTreeDumpFile.substring(0,lastDotIndex);
      }
      else
      {
        extension = "";
        stem = mTreeDumpFile;
      }
      String thisTurnDumpFile = stem + "." + mOurRole + "." + (mRoot.getDepth()/mNumRoles) + extension;
      mRoot.dumpTree(thisTurnDumpFile);
    }

    assert(bestMoveInfo != null) : "Best move choice info was illegally null";
    return bestMoveInfo;
  }

  int[] mResponseMoveMasterIndex = null;

  private void calculateCriticalResponses()
  {
    IPlayoutPolicy playoutPolicy = mUnderlyingStateMachine.getPlayoutPolicy();
    if ( playoutPolicy != null && playoutPolicy instanceof PlayoutPolicyCriticalResponse)
    {
      ForwardDeadReckonLegalMoveInfo[] masterMoveList = mUnderlyingStateMachine.getFullPropNet().getMasterMoveList();
      if ( mResponseMoveMasterIndex == null )
      {
        mResponseMoveMasterIndex = new int[masterMoveList.length];
        for(int i = 0; i < mResponseMoveMasterIndex.length; i++)
        {
          mResponseMoveMasterIndex[i] = -1;
        }

        ((PlayoutPolicyCriticalResponse)playoutPolicy).setReponseMap(mResponseMoveMasterIndex);
      }

      TreeNode rootNode = mRoot;
      while(rootNode.mNumChildren == 1)
      {
        if ( !(rootNode.mChildren[0] instanceof TreeEdge) )
        {
          return;
        }
        rootNode = rootNode.get(((TreeEdge)rootNode.mChildren[0]).getChildRef());
      }

      int choosingRawRoleIndex = mUnderlyingStateMachine.getRoleOrdering().roleIndexToRawRoleIndex((rootNode.mDecidingRoleIndex+1)%mNumRoles);
      for(int i = 0; i < mResponseMoveMasterIndex.length; i++)
      {
        if ( masterMoveList[i].mRoleIndex == choosingRawRoleIndex )
        {
          mResponseMoveMasterIndex[i] = -1;
        }
      }

      Map<Move,Map<Move,Double> > criticalMoveMap = new HashMap<>();

      for(int i = 0; i < rootNode.mNumChildren; i++)
      {
        Object choice = rootNode.mChildren[i];

        if ( choice instanceof TreeEdge && ((TreeEdge)choice).getChildRef() != TreeNode.NULL_REF)
        {
          TreeNode child = rootNode.get(((TreeEdge)choice).getChildRef());
          if ( child != null )
          {
            Map<Move, Double> moveMap = new HashMap<>();
            criticalMoveMap.put(((TreeEdge)choice).mPartialMove.mMove, moveMap);
            //LOGGER.info("Critical responses to " + ((TreeEdge)choice).mPartialMove.move + ":");

            double scoreAccumulator = 0;
            double scoreSquareAccumulator = 0;
            int count = 0;

            for(int j = 0; j < child.mNumChildren; j++)
            {
              Object childChoice = child.mChildren[j];
              if ( childChoice instanceof TreeEdge && ((TreeEdge)childChoice).getChildRef() != TreeNode.NULL_REF)
              {
                TreeNode grandChild = rootNode.get(((TreeEdge)childChoice).getChildRef());
                if ( grandChild != null )
                {
                  double score = grandChild.getAverageScore(grandChild.mDecidingRoleIndex);
                  scoreAccumulator += score;
                  scoreSquareAccumulator += score*score;
                  count++;
                }
              }
            }

            //LOGGER.info("Calculating critical choices with consider count of " + count);
            if ( count > 0 )
            {
              scoreAccumulator /= count;
              scoreSquareAccumulator /= count;

              double stdDevScore = Math.sqrt(scoreSquareAccumulator - scoreAccumulator*scoreAccumulator);

              for(int j = 0; j < child.mNumChildren; j++)
              {
                Object childChoice = child.mChildren[j];
                if ( childChoice instanceof TreeEdge && ((TreeEdge)childChoice).getChildRef() != TreeNode.NULL_REF)
                {
                  TreeNode grandChild = rootNode.get(((TreeEdge)childChoice).getChildRef());
                  if ( grandChild != null && (grandChild.mNumVisits > 10 || grandChild.mComplete) )
                  {
                    double score = grandChild.getAverageScore(grandChild.mDecidingRoleIndex);

                    moveMap.put(((TreeEdge)childChoice).mPartialMove.mMove, (score - scoreAccumulator)/stdDevScore);
  //                  if ( score > scoreAccumulator + stdDevScore )
  //                  {
  //                    LOGGER.info("    " + ((TreeEdge)childChoice).mPartialMove.move + "scores " + score);
  //                  }
                  }
                }
              }
            }
          }
        }
      }

      for(int i = 0; i < rootNode.mNumChildren; i++)
      {
        Object choice = rootNode.mChildren[i];

        if ( choice instanceof TreeEdge )
        {
          Move move = ((TreeEdge)choice).mPartialMove.mMove;
          Map<Move,Double> moveMap = criticalMoveMap.get(move);
          double bestResponseScore = 0;
          Move bestResponse = null;
          assert(((TreeEdge)choice).mPartialMove.mRoleIndex == choosingRawRoleIndex);

          for(Move response : moveMap.keySet())
          {
            Double score = moveMap.get(response);

            if ( score != null && score > 0.9 )
            {
              Map<Move,Double> responseMoveMap = criticalMoveMap.get(response);
              if ( responseMoveMap != null )
              {
                Double swapScore = responseMoveMap.get(move);
                if ( swapScore != null && swapScore > 0.98 )
                {
                  double aggregateScore = score + swapScore;
                  if ( aggregateScore > bestResponseScore )
                  {
                    bestResponseScore = aggregateScore;
                    bestResponse = response;
                  }
                }
              }
            }
          }

          if ( bestResponse != null && bestResponseScore > 2 )
          {
            int responseIndex = -1;
            for(int j = 0; j < masterMoveList.length; j++)
            {
              if ( masterMoveList[j].mRoleIndex != choosingRawRoleIndex && masterMoveList[j].mMove.equals(bestResponse))
              {
                responseIndex = j;
                break;
              }
            }
            mResponseMoveMasterIndex[((TreeEdge)choice).mPartialMove.mMasterIndex] = responseIndex;
            //LOGGER.info("Best response to " + move + " is " + bestResponse + " with criticality score " + bestResponseScore);
          }
        }
      }

      for(int j = 0; j < masterMoveList.length; j++)
      {
        if ( mResponseMoveMasterIndex[j] != -1 )
        {
          LOGGER.info("Best response to " + masterMoveList[j].mInputProposition + " is " + masterMoveList[mResponseMoveMasterIndex[j]].mInputProposition);
        }
      }
    }
  }

  void validateAll()
  {
    if (mRoot != null)
      mRoot.validate(true);

    for (ForwardDeadReckonInternalMachineState e : mPositions.keySet())
    {
      TreeNode node = findTransposition(e);

      if (node != null)
      {
        if (node.mDecidingRoleIndex != mNumRoles - 1)
        {
          LOGGER.warn("Position references bad type");
        }
        if (!node.mState.equals(e))
        {
          LOGGER.warn("Position state mismatch");
        }
      }
    }

/*
    for (Object lNodeAsObject : mNodePool.getItemTable())
    {
      TreeNode lNode = (TreeNode)lNodeAsObject;
      if (lNode != null && !lNode.mFreed)
      {
        if (lNode.mDecidingRoleIndex == mNumRoles - 1)
        {
          if (lNode != findTransposition(lNode.mState))
          {
            LOGGER.warn("Missing reference in positions table");
            LOGGER.warn("node state is: " + lNode.mState + " with hash " + lNode.mState.hashCode());
            LOGGER.warn(findTransposition(lNode.mState));
          }
        }
      }
    }
*/
  }

  /**
   * Set the forced moves for each role whose move IS forced in the specified state and we
   * are eliminating non-decision nodes
   * @param xiState
   * @param jointMove
   * @return true if all roles had forced moves
   */
  public boolean setForcedMoveProps(ForwardDeadReckonInternalMachineState xiState, ForwardDeadReckonLegalMoveInfo[] jointMove)
  {
    if (mRemoveNonDecisionNodes)
    {
      boolean result = true;

      for(int i = 0; i < mNumRoles; i++)
      {
        Role role = mRoleOrdering.roleIndexToRole(i);
        ForwardDeadReckonLegalMoveSet moves = mUnderlyingStateMachine.getLegalMoveSet(xiState);
        if (mSearchFilter.getFilteredMovesSize(xiState, moves, role, true) == 1)
        {
          jointMove[i] = moves.getContents(role).iterator().next();
        }
        else
        {
          result = false;
        }
      }

      return result;
    }

    return false;
  }

  public boolean validateForcedMoveProps(ForwardDeadReckonInternalMachineState xiState, ForwardDeadReckonLegalMoveInfo[] jointMove)
  {
    if (mRemoveNonDecisionNodes)
    {
      for(int i = 0; i < mNumRoles; i++)
      {
        Role role = mRoleOrdering.roleIndexToRole(i);
        ForwardDeadReckonLegalMoveSet moves = mUnderlyingStateMachine.getLegalMoveSet(xiState);
        if (mSearchFilter.getFilteredMovesSize(xiState, moves, role, true) == 1)
        {
          ForwardDeadReckonLegalMoveInfo expectedMove = moves.getContents(role).iterator().next();
          if ((jointMove[i].mInputProposition == null) != (expectedMove.mInputProposition == null))
          {
            return false;
          }
        }
      }
    }

    return true;
  }

  private void selectAction(boolean forceSynchronous, Move xiChosenMove)
    throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException
  {
    //completedNodeQueue.clear();

    long lSelectStartTime = System.nanoTime();
    TreePath visited = mPathPool.allocate(mTreePathAllocator);
    TreeNode cur = mRoot;
    TreePathElement selected = null;
    int parentDepth = 0;

    while (!cur.isUnexpanded())
    {
      //  Hyper expand first choice layer for each role.
      if ( mAllowHyperExpansion && cur.getDepth() < mRoot.getDepth()+2*mNumRoles && cur.mNumChildren > 1)
      {
        setForcedMoveProps(cur.mState, mJointMoveBuffer);
        cur.hyperExpand(visited, mJointMoveBuffer, MAX_HYPER_RECURSION_DEPTH);
      }

      parentDepth = cur.getDepth();
      selected = cur.select(visited, mJointMoveBuffer, xiChosenMove);
      cur = selected.getChildNode();
      xiChosenMove = null;
    }

    long lExpandStartTime = System.nanoTime();
    TreeNode newNode;
    if (!cur.mComplete && (cur.mNumVisits >= mExpandAfterVisitThreshold || cur.getDepth() >= mShallowestCompletionDepth - mNumRoles))
    {
      assert(selected == null || cur == selected.getChildNode());
      assert(selected == null || cur.mParents.contains(selected.getParentNode()));

      //  Expand for each role so we're back to our-move as we always rollout after joint moves
      cur = cur.expand(visited, mJointMoveBuffer, parentDepth);
      assert(selected == null || cur == selected.getChildNode());
    }


    //  Even if we've selected a terminal node we still do a pseudo-rollout
    //  from it so its value gets a weight increase via back propagation
    newNode = cur;

    long selectTime = lExpandStartTime - lSelectStartTime;
    long expandTime = System.nanoTime() - lExpandStartTime;

    if (selectTime > maxSelectTime)
    {
      maxSelectTime = selectTime;
      LOGGER.debug("Max select time seen (ms): " + (selectTime/1000000));
    }
    if (expandTime > maxExpandTime)
    {
      maxExpandTime = expandTime;
      LOGGER.debug("Max expand time seen (ms): " + (expandTime/1000000));
    }

    // Perform the rollout request.
    assert(selected != null || newNode == mRoot);
    assert(selected == null || newNode == selected.getChildNode());
    assert(!newNode.mFreed);
    newNode.rollOut(visited,
                    mGameSearcher.getPipeline(),
                    forceSynchronous,
                    selectTime,
                    expandTime);
  }

  /**
   * Permanently store the values associate with a state.
   *
   * @param xiState         - the state.
   * @param xiTerminal      - whether the state is terminal.
   * @param xiComplete      - whether the state is complete.
   * @param xiNumVisits     - the number of times the state has been visited.
   * @param xiAverageScore  - the average score for the state.
   */
  public void storeState(ForwardDeadReckonInternalMachineState xiState,
                         boolean xiTerminal,
                         boolean xiComplete,
                         int xiNumVisits,
                         double xiAverageScore)
  {
    assert(CREATING_DATABASE);
    long[] lBits = xiState.getContents().getBits();

    if ((xiTerminal) || (xiComplete) || (xiNumVisits > 1000))
    {
      try
      {
        mDatabase.writeBoolean(xiTerminal);
        mDatabase.writeBoolean(xiComplete);
        mDatabase.writeInt(xiNumVisits);
        mDatabase.writeDouble(xiAverageScore);
        mDatabase.writeInt(lBits.length);
        for (long lLong : lBits)
        {
          mDatabase.writeLong(lLong);
        }
      }
      catch (IOException lEx) { /* Oh well. */ }
    }
  }
}
