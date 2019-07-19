package org.ggp.base.player.gamer.statemachine.sancho;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.ThreadContext;
import org.ggp.base.player.gamer.event.GamerSelectedMoveEvent;
import org.ggp.base.player.gamer.statemachine.sample.SampleGamer;
import org.ggp.base.player.gamer.statemachine.sancho.MachineSpecificConfiguration.CfgItem;
import org.ggp.base.player.gamer.statemachine.sancho.PayoffMatrixGamePlayer.UnsupportedGameException;
import org.ggp.base.player.gamer.statemachine.sancho.Watchdog.WatchdogExpiryHandler;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.CombinedHeuristic;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.GoalsStabilityHeuristic;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.MajorityGoalsHeuristic;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.PieceHeuristic;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.propnet.polymorphic.PolymorphicProposition;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropositionInfo;
import org.ggp.base.util.statemachine.MachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.StateMachine;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.Factor;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine.PlayoutInfo;
import org.ggp.base.util.statemachine.playoutPolicy.PlayoutPolicyGoalGreedyWithPop;
import org.ggp.base.util.symbol.grammar.SymbolPool;

import com.google.common.io.CharStreams;

/**
 * The Sancho General Game Player.
 */
public class Sancho extends SampleGamer implements WatchdogExpiryHandler
{
  private static final Logger LOGGER = LogManager.getLogger();
  private static final long SAFETY_MARGIN = MachineSpecificConfiguration.getCfgInt(CfgItem.SAFETY_MARGIN);
  private static final int MIN_PRIMARY_SIMULATION_SAMPLES = 100;

  // Do one-off start-of-day processing.
  static
  {
    StdOutErrLog.tieSystemOutAndErrToLog();
  }
  private static final String LAST_COMMIT = getWorkingCopyRevision();
  private static final boolean ASSERTIONS_ENABLED = areAssertionsEnabled();

  /**
   * When adding additional state, you MUST null out references in {@link #tidyUp()}.
   */
  private Role                                  mOurRole;
  private int                                   mTurn                            = 0;
  private String                                mPlanString                      = null;
  private GamePlan                              mPlan                            = null;
  private RoleOrdering                          mRoleOrdering                    = null;
  private ForwardDeadReckonPropositionInfo[]    mRoleControlProps                = null;
  private ForwardDeadReckonPropnetStateMachine  mUnderlyingStateMachine          = null;
  private ForwardDeadReckonInternalMachineState mPreviousTurnRootState           = null;
  private StateMachineProxy                     mStateMachineProxy               = null;
  private int                                   mNumRoles                        = 0;
  private int                                   mMinRawNetScore                  = 0;
  private int                                   mMaxRawNetScore                  = 100;
  private int                                   mMultiRoleAverageScoreDiff       = 0;
  private short                                 mCurrentMoveDepth                = 0;
  private boolean                               mUnderExpectedRangeScoreReported = false;
  private boolean                               mOverExpectedRangeScoreReported  = false;
  private GameSearcher                          mSearchProcessor                 = null;
  private String                                mLogName                         = null;
  private SystemStatsLogger                     mSysStatsLogger                  = null;
  private Move                                  mLastMove                        = null;
  private ForwardDeadReckonLegalMoveInfo        mLastMoveInfo                    = null;
  private int                                   mFinalScore                      = -1;
  private boolean                               mSolvedFromStart                 = false;
  private Tlkio                                 mBroadcaster                     = null;
  private Watchdog                              mWatchdog                        = null;
  public boolean                                mUTLearnCharacteristicsOnly      = false;
  /**
   * When adding additional state, you MUST null out references in {@link #tidyUp()}.
   */

  @Override
  public void configure(int xiParamIndex, String xiParam)
  {
    // At the moment, Sancho can only be configured with a "plan" of initial
    // moves that it should play - used for testing.
    if (!xiParam.startsWith("plan="))
    {
      throw new InvalidParameterException();
    }
    mPlanString = xiParam.substring(5);

    // Start pre-warming the search tree 2 moves before he end of the plan so that we arrive at the turn we need to
    // search in earnest in with a warmed-up tree.  (This isn't needed in the case of a learned plan, because they
    // always take us to the end of the game.)
    GameSearcher.thinkBelowPlanSize = 2;
  }

  private boolean isUsingConfiguredPlan()
  {
    return mPlanString != null;
  }

  private int netScore(ForwardDeadReckonPropnetStateMachine stateMachine,
                       ForwardDeadReckonInternalMachineState state)
  {
    int result = 0;
    int bestEnemyScore = 0;
    for (Role role : stateMachine.getRoles())
    {
      if (!role.equals(mOurRole))
      {
        int score = stateMachine.getGoal(state, role);
        if (score > bestEnemyScore)
        {
          bestEnemyScore = score;
        }
      }
      else
      {
        result = stateMachine.getGoal(state, role);
      }
    }

    int winBonus = 0;
    if (result >= bestEnemyScore)
    {
      winBonus += 5;

      if (result > bestEnemyScore)
      {
        winBonus += 5;
      }
    }
    int rawResult = (mGameCharacteristics.numRoles == 1 ? result : ((result + winBonus) * 100) / 110);
    int normalizedResult = ((rawResult - mMinRawNetScore) * 100) /
                           (mMaxRawNetScore - mMinRawNetScore);

    if (normalizedResult > 100 && !mOverExpectedRangeScoreReported)
    {
      normalizedResult = 100;
      mOverExpectedRangeScoreReported = true;
      LOGGER.warn("Saw score that nornmalized to > 100");
    }
    else if (normalizedResult < 0 && !mUnderExpectedRangeScoreReported)
    {
      normalizedResult = 0;
      mUnderExpectedRangeScoreReported = true;
      LOGGER.warn("Saw score that nornmalized to < 0");
    }

    return normalizedResult;
  }

  @Override
  public String getName()
  {
    return MachineSpecificConfiguration.getCfgStr(CfgItem.PLAYER_NAME);
  }

  @Override
  public StateMachine getInitialStateMachine()
  {
    String lMatchID = getMatch().getMatchId();
    mLogName = lMatchID + "-" + getPort();
    ThreadContext.put("matchID", mLogName);

    // If we still have a watchdog from a previous game, it must be running as a zombie.  (The framework prevents us
    // from playing a new game if it thinks we're still running an old one.)  Immediately tidy-up the old match.
    if (mWatchdog != null)
    {
      LOGGER.error("Watchdog exists at start of new match");
      tidyUp(false, true);
    }

    mWatchdog = new Watchdog(MachineSpecificConfiguration.getCfgInt(CfgItem.DEAD_MATCH_INTERVAL), this);

    ThreadControl.sCPUIdParity = (getPort()%2 == 0);
    ThreadControl.reset();

    mUnderlyingStateMachine = new ForwardDeadReckonPropnetStateMachine(ThreadControl.CPU_INTENSIVE_THREADS + 1,
                                                                      getMetaGamingTimeout(),
                                                                      getRole(),
                                                                      mGameCharacteristics);
    System.gc();

    mCurrentMoveDepth = 0;

    mStateMachineProxy = new StateMachineProxy(mUnderlyingStateMachine);
    return mStateMachineProxy;
  }

  private static int unNormalizedStateDistance(MachineState queriedState, MachineState targetState)
  {
    int matchCount = 0;

    for (GdlSentence s : targetState.getContents())
    {
      if (queriedState.getContents().contains(s))
      {
        matchCount++;
      }
    }

    return targetState.getContents().size() - matchCount;
  }

  /**
   * @return the commit ID of the latest revision in the working copy.
   */
  private static String getWorkingCopyRevision()
  {
    String lLastCommit = "<Unknown>";

    String lWCPath = MachineSpecificConfiguration.getCfgStr(CfgItem.WC_LOCATION);
    if (lWCPath != null)
    {
      try
      {
        File lWCRoot = new File(lWCPath);
        InputStream lInput = Runtime.getRuntime().exec("git rev-parse HEAD", null, lWCRoot).getInputStream();
        lLastCommit = CharStreams.toString(new InputStreamReader(lInput));
        lLastCommit = lLastCommit.replace("\r", "").replace("\n", "");
      }
      catch (IOException lEx)
      {
        LOGGER.warn("Failed to get git version", lEx);
      }
    }

    return lLastCommit;
  }

  private static boolean areAssertionsEnabled()
  {
    boolean lAssertionsEnabled = false;
    assert ((lAssertionsEnabled = true) == true);

    if (lAssertionsEnabled)
    {
      System.err.println("WARNING: Assertions are enabled - this will impact performance");
    }

    return lAssertionsEnabled;
  }

  @Override
  public void stateMachineMetaGame(long timeout)
     throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException
  {
    LOGGER.info("This is " + getName() + " with last commit " + LAST_COMMIT);
    MachineSpecificConfiguration.logConfig();
    if (ASSERTIONS_ENABLED)
    {
      LOGGER.warn("WARNING: Assertions are enabled - this will impact performance");
    }

    mSysStatsLogger = new SystemStatsLogger(mLogName);

    mSearchProcessor = new GameSearcher(MachineSpecificConfiguration.getCfgInt(CfgItem.NODE_TABLE_SIZE),
                                        mUnderlyingStateMachine.getRoles().length,
                                        mLogName);
    mStateMachineProxy.setController(mSearchProcessor);

    mPreviousTurnRootState = null; //  No move seen yet

    if (!ThreadControl.RUN_SYNCHRONOUSLY)
    {
      Thread lSearchProcessorThread = new Thread(mSearchProcessor, "Search Processor");
      lSearchProcessorThread.setDaemon(true);
      lSearchProcessorThread.start();
    }

    Random r = new Random();

    if ( MachineSpecificConfiguration.getCfgStr(CfgItem.TLKIO_CHANNEL) != MachineSpecificConfiguration.NO_TLK_CHANNEL )
    {
      mBroadcaster = new Tlkio(getName(), MachineSpecificConfiguration.getCfgStr(CfgItem.TLKIO_CHANNEL));
      mSearchProcessor.setBroadcaster(mBroadcaster);
    }

    // If have been configured with a plan (for test purposes), load it now.
    // We'll still do everything else as normal, but whilst there are moves in
    // the plan, when it comes to play, we'll just play the specified move.
    mPlan = new GamePlan();
    if (mPlanString != null)
    {
      mPlan.considerPlan(convertPlanString(mPlanString));

      //  When given an explicit plan at the start of the game (implies we're in test mode)
      //  warm up the search tree 2 moves before the plan runs out
      GameSearcher.thinkBelowPlanSize = 2;
    }
    else
    {
      GameSearcher.thinkBelowPlanSize = 0;
    }

    mOurRole = getRole();
    LOGGER.info("Start clock: " + getMatch().getStartClock() + "s");
    LOGGER.info("Play clock:  " + getMatch().getPlayClock() + "s");
    LOGGER.info("We are:      " + mOurRole);

    mNumRoles = mUnderlyingStateMachine.getRoles().length;
    mRoleOrdering = mUnderlyingStateMachine.getRoleOrdering();

    mTurn = 0;
    StatsLogUtils.Series.TURN.logDataPoint(System.currentTimeMillis(), mTurn);

    mSolvedFromStart = false;
    mMinRawNetScore = 0;
    mMaxRawNetScore = 100;
    mUnderExpectedRangeScoreReported = false;
    mOverExpectedRangeScoreReported = false;

    int observedMinNetScore = Integer.MAX_VALUE;
    int observedMaxNetScore = Integer.MIN_VALUE;
    long simulationsPerformed = 0;
    int multiRoleSamples = 0;
    boolean greedyRolloutsDisabled = MachineSpecificConfiguration.getCfgBool(CfgItem.DISABLE_GREEDY_ROLLOUTS);

    mMultiRoleAverageScoreDiff = 0;

    // Set up those game characteristics that we only know now that we've got a state machine.
    mGameCharacteristics.numRoles = mNumRoles;

    // Check if we already know how to solve this game.
    String lSavedPlan = mGameCharacteristics.getPlan();
    makePreMatchAnnouncement(lSavedPlan);
    if (lSavedPlan != null)
    {
      // We've played this game before and know how to solve it.
      LOGGER.info("Considering saved plan: " + lSavedPlan);
      mPlan.considerPlan(convertPlanString(mGameCharacteristics.getPlan()));
      mSolvedFromStart = true;
      LOGGER.info("Ready to play");
      return;
    }

    // Analyze game semantics.  This includes latch identification and search filter generation.
    long lSemanticAnalysisStartTime = System.currentTimeMillis();
    long lSemanticAnalysisStopTime = Math.min(timeout - 10000,
                                              lSemanticAnalysisStartTime + (timeout - lSemanticAnalysisStartTime) / 2);

    mUnderlyingStateMachine.performSemanticAnalysis(lSemanticAnalysisStopTime);

    PayoffMatrixGamePlayer lPayoffMatrixGamePlayer;
    try
    {
      lPayoffMatrixGamePlayer = new PayoffMatrixGamePlayer(mUnderlyingStateMachine, lSemanticAnalysisStopTime);
      LOGGER.info("Game can be represented by a simple payoff matrix");
    }
    catch (UnsupportedGameException lEx) {/* Do nothing */}

    CombinedHeuristic heuristic;

    MajorityGoalsHeuristic goalsPredictionHeuristic = new MajorityGoalsHeuristic();
    GoalsStabilityHeuristic goalsStabilityHeuristic = new GoalsStabilityHeuristic();
    PieceHeuristic pieceHeuristic = new PieceHeuristic();

    if (MachineSpecificConfiguration.getCfgBool(CfgItem.DISABLE_PIECE_HEURISTIC))
    {
      heuristic = new CombinedHeuristic(goalsPredictionHeuristic, goalsStabilityHeuristic);
    }
    else
    {
      heuristic = new CombinedHeuristic(pieceHeuristic, goalsPredictionHeuristic, goalsStabilityHeuristic);
    }

    boolean hasHeuristicCandidates = heuristic.tuningInitialise(mUnderlyingStateMachine, mRoleOrdering);

    ForwardDeadReckonInternalMachineState initialState = mUnderlyingStateMachine.createInternalState(getCurrentState());

    //	Sample to see if multiple roles have multiple moves available
    //	implying this must be a simultaneous move game
    //	HACK - actually only count games where both players can play the
    //	SAME move - this gets blocker but doesn't include fully factored
    //	games like C4-simultaneous or Chinook (but it's a hack!)
    mGameCharacteristics.isSimultaneousMove = false;
    mGameCharacteristics.isPseudoSimultaneousMove = false;
    mGameCharacteristics.isPseudoPuzzle = mUnderlyingStateMachine.getIsPseudoPuzzle();
    mGameCharacteristics.isStrictlyAlternatingPlay = true;

    //  Create masks of possible control props, which we'll whittle down during simulation
    //  If we wind up with a unique prop for each role e'll note it for future use
    ForwardDeadReckonInternalMachineState[] roleControlMasks = new ForwardDeadReckonInternalMachineState[mNumRoles];
    mRoleControlProps = new ForwardDeadReckonPropositionInfo[mNumRoles];

    for(int i = 0; i < mNumRoles; i++)
    {
      roleControlMasks[i] = mUnderlyingStateMachine.createEmptyInternalState();
      roleControlMasks[i].clear();
      roleControlMasks[i].invert();
    }

    //	Also monitor whether any given player always has the SAME choice of move (or just a single choice)
    //	every turn - such games are (highly probably) iterated games
    List<Set<Move>> roleMoves = new ArrayList<>();
    mGameCharacteristics.isIteratedGame = true;

    for (int i = 0; i < mNumRoles; i++)
    {
      roleMoves.add(null);
    }

    double branchingFactorApproximation = 0;
    double averageHyperSequenceLength = 0;
    double varianceHyperSequenceLength = 0;
    int[] roleScores = new int[mNumRoles];

    Collection<Factor> factors = mUnderlyingStateMachine.getFactors();

    //	Perform a small number of move-by-move simulations to assess how
    //	the potential piece count heuristics behave at the granularity of
    //	a single decision
    long lMetaGameStartTime = System.currentTimeMillis();
    long lMetaGameStopTime = timeout - 5000;
    int numSamples = 0;
    int maxNumTurns = 0;
    int minNumTurns = Integer.MAX_VALUE;

    // Spend half the time determining heuristic weights if there are any heuristics, else spend
    //  a short time just establishing the type of game
    long lHeuristicStopTime;

    if (hasHeuristicCandidates)
    {
      lHeuristicStopTime = (lMetaGameStartTime + lMetaGameStopTime) / 2;
    }
    else
    {
      lHeuristicStopTime = lMetaGameStartTime + 4000;
    }

    //  Buffer for new states
    ForwardDeadReckonInternalMachineState newState = new ForwardDeadReckonInternalMachineState(initialState);
    int[] playerMoveParity = new int[mNumRoles];
    for(int i = 0; i < mNumRoles; i++)
    {
      playerMoveParity[i] = -1;
    }

    //  Start with greedy rollouts disabled for finding basic characteristics.  This is likely to result in more
    //  simulations, and prevents games with forced greedy paths (like Centipede wherein one choice always leads
    //  to immediate non-win termination and hence is never chosen by greedy processing) giving misleading results
    mUnderlyingStateMachine.enableGreedyRollouts(false, false);

    int totalTurnSamples = 0;
    int totalGoalChangeCount = 0;
    int totalHyperSequenceLength = 0;
    int totalSquaredHyperSequenceLength = 0;
    int totalHyperSequenceCount = 0;
    int totalMoveChoices = 0;
    boolean goalsMonotonic = true;

    //  Slight hack, but for now we don't bother continuing to simulate for a long time after discovering we're in
    //  a simultaneous turn game, because (for now anyway) we disable heuristics in such games anyway
    while ((System.currentTimeMillis() < lHeuristicStopTime) &&
           ((numSamples < MIN_PRIMARY_SIMULATION_SAMPLES) || (!mGameCharacteristics.isSimultaneousMove)))
    {
      ForwardDeadReckonInternalMachineState sampleState = new ForwardDeadReckonInternalMachineState(initialState);

      int numRoleMovesSimulated = 0;
      int numBranchesTaken = 0;
      int turnNum = 0;
      int lastOurGoal = mUnderlyingStateMachine.getGoal(sampleState, mOurRole);
      int hyperExpansionLength = -1;
      int previousChoosingRoleIndex;
      int choosingRoleIndex = -2;

      heuristic.tuningStartSampleGame();

      while (!mUnderlyingStateMachine.isTerminal(sampleState))
      {
        boolean roleWithChoiceSeen = false;
        ForwardDeadReckonLegalMoveInfo[] jointMove = new ForwardDeadReckonLegalMoveInfo[mNumRoles];
        Set<Move> allMovesInState = new HashSet<>();

        previousChoosingRoleIndex = choosingRoleIndex;
        choosingRoleIndex = -2;

        turnNum++;
        hyperExpansionLength++;

        for (int i = 0; i < mNumRoles; i++)
        {
          List<ForwardDeadReckonLegalMoveInfo> legalMoves =
                  new ArrayList<>(mUnderlyingStateMachine.getLegalMoves(sampleState, mRoleOrdering.roleIndexToRole(i)));

          totalMoveChoices += legalMoves.size();
          if (legalMoves.size() > 1)
          {
            if (roleControlMasks[i].size() > 0)
            {
              if (roleControlMasks[i].intersectionSize(sampleState) == 0)
              {
                LOGGER.debug("Eliminating role control props");
              }
              //  This player has control (may not be only this player)
              roleControlMasks[i].intersect(sampleState);
            }

            Set<Move> previousChoices = roleMoves.get(i);

            if (previousChoices != null)
            {
              if (mGameCharacteristics.isIteratedGame)
              {
                Set<Move> moves = new HashSet<>();

                for(ForwardDeadReckonLegalMoveInfo moveInfo : legalMoves)
                {
                  moves.add(moveInfo.mMove);
                }

                if (!previousChoices.equals(moves))
                {
                  mGameCharacteristics.isIteratedGame = false;
                }
              }
            }
            else
            {
              Set<Move> moves = new HashSet<>();

              for(ForwardDeadReckonLegalMoveInfo moveInfo : legalMoves)
              {
                moves.add(moveInfo.mMove);
              }

              roleMoves.set(i, moves);
            }

            choosingRoleIndex = i;
            Factor turnFactor = null;

            if (mGameCharacteristics.isStrictlyAlternatingPlay)
            {
              int moveParity = (turnNum%mNumRoles);
              if (playerMoveParity[choosingRoleIndex] == -1)
              {
                playerMoveParity[choosingRoleIndex] = moveParity;
              }
              else if (moveParity != playerMoveParity[choosingRoleIndex])
              {
                mGameCharacteristics.isStrictlyAlternatingPlay = false;
              }
            }

            for (ForwardDeadReckonLegalMoveInfo moveInfo : legalMoves)
            {
              if (factors != null)
              {
                for(Factor factor : factors)
                {
                  if (factor.getMoves().contains(moveInfo.mMove))
                  {
                    if (turnFactor != null && turnFactor != factor)
                    {
                      mGameCharacteristics.moveChoicesFromMultipleFactors = true;
                      break;
                    }
                    turnFactor = factor;
                  }
                }
              }
              if (allMovesInState.contains(moveInfo.mMove))
              {
                mGameCharacteristics.isSimultaneousMove = true;
                choosingRoleIndex = -1;
                break;
              }
              allMovesInState.add(moveInfo.mMove);
            }

            if (roleWithChoiceSeen)
            {
              mGameCharacteristics.isPseudoSimultaneousMove = true;
              choosingRoleIndex = -1;
            }

            roleWithChoiceSeen = true;

            numBranchesTaken += legalMoves.size();
            numRoleMovesSimulated++;
          }

          jointMove[i] = legalMoves.get(r.nextInt(legalMoves.size()));
        }

        if (choosingRoleIndex < 0 ||
            (choosingRoleIndex != previousChoosingRoleIndex && previousChoosingRoleIndex >= 0))
        {
          totalHyperSequenceLength += hyperExpansionLength;
          totalSquaredHyperSequenceLength += hyperExpansionLength*hyperExpansionLength;
          hyperExpansionLength = 0;
          totalHyperSequenceCount++;
        }

        heuristic.tuningInterimStateSample(sampleState, choosingRoleIndex);

        mUnderlyingStateMachine.getNextState(sampleState, null, jointMove, newState);

        sampleState.copy(newState);

        int currentOurGoal = mUnderlyingStateMachine.getGoal(sampleState, mOurRole);

        totalTurnSamples++;
        if (currentOurGoal != lastOurGoal)
        {
          totalGoalChangeCount++;
          if (lastOurGoal > currentOurGoal)
          {
            goalsMonotonic = false;
          }
          lastOurGoal = currentOurGoal;
        }
      }

      if (hyperExpansionLength > 0)
      {
        totalHyperSequenceLength += hyperExpansionLength;
        totalSquaredHyperSequenceLength += hyperExpansionLength*hyperExpansionLength;
        totalHyperSequenceCount++;
      }

      if (turnNum > maxNumTurns)
      {
        maxNumTurns = turnNum;
      }
      if (turnNum < minNumTurns)
      {
        assert(turnNum>0);
        minNumTurns = turnNum;
      }

      for (int i = 0; i < mNumRoles; i++)
      {
        roleScores[i] = mUnderlyingStateMachine.getGoal(sampleState, mRoleOrdering.roleIndexToRole(i));
      }

      // Tell the heuristic about the terminal state, for tuning purposes.
      assert(mUnderlyingStateMachine.isTerminal(sampleState));
      heuristic.tuningTerminalStateSample(sampleState, roleScores);

      if (numRoleMovesSimulated > 0)
      {
        branchingFactorApproximation += (numBranchesTaken / numRoleMovesSimulated);
      }
      numSamples++;
    }

    // Complete heuristic tuning.
    heuristic.tuningComplete();

    //  If we were able to run very few samples only don't make non-default
    //  assumptions about the game based on the inadequate sampling
    LOGGER.info("Performed " + numSamples + " simulations to calculate heuristics");
    if (numSamples < MIN_PRIMARY_SIMULATION_SAMPLES)
    {
      mGameCharacteristics.isIteratedGame = false;
      heuristic.pruneAll();

      LOGGER.warn("Insufficient sampling time to reliably ascertain game characteristics");
    }
    else
    {
      mGameCharacteristics.hasAdequateSampling = true;
      averageHyperSequenceLength = (double)totalHyperSequenceLength/totalHyperSequenceCount;
      varianceHyperSequenceLength = (double)totalSquaredHyperSequenceLength/totalHyperSequenceCount -
                                    averageHyperSequenceLength * averageHyperSequenceLength;
      branchingFactorApproximation /= numSamples;
    }

    //  For now we don't attempt heuristic usage in simultaneous move games
    if (mGameCharacteristics.isSimultaneousMove)
    {
      heuristic.pruneAll();
    }

    //  If we saw differing game lengths do not treat as an iterated game.  Note that we have to do this
    //  here, and not after the more discriminatory simulation loop later, because the later loop
    //  also tests greedy rollout efficacy, and this can force move paths that give a misleading result for
    //  for the possible range of game lengths.
    //  Similarly we do not consider anything with 4 or less turns as a candidate for iterated analysis
    if (maxNumTurns != minNumTurns || maxNumTurns < 5)
    {
      mGameCharacteristics.isIteratedGame = false;
    }


    mGameCharacteristics.setGoalsStability(goalsStabilityHeuristic.getGoalStability());

    if (mGameCharacteristics.isSimultaneousMove || mGameCharacteristics.isPseudoSimultaneousMove)
    {
      mRoleControlProps = null;

      greedyRolloutsDisabled = true;
      mUnderlyingStateMachine.enableGreedyRollouts(false, true);
    }
    else
    {
      //  Turn greedy rollouts back on for now to measure their effectiveness
      mUnderlyingStateMachine.enableGreedyRollouts(true, false);

      if (mNumRoles == 1)
      {
        mRoleControlProps = null;
      }
      else
      {
        //  Did we identify unique control props?
        //  We look for a proposition that is always true when a given role has more than one
        //  move choice, and is never true if any other player has more than one move choice
        //  This is slightly ovr-specific, and will not work for games with control denoted by a
        //  set of props rather than a single one (e.g. - Pentago), but it will suffice for now
        ForwardDeadReckonInternalMachineState[] inverseMasks = new ForwardDeadReckonInternalMachineState[mNumRoles];

        for(int i = 0; i < mNumRoles; i++)
        {
          inverseMasks[i] = new ForwardDeadReckonInternalMachineState(roleControlMasks[i]);
          inverseMasks[i].invert();
        }

        for(int i = 0; i < mNumRoles; i++)
        {
          //  Eliminate anything common to multiple roles
          for(int j = 0; j < mNumRoles; j++)
          {
            if (j != i)
            {
              roleControlMasks[i].intersect(inverseMasks[j]);
            }
          }
          if (roleControlMasks[i].size() != 1)
          {
            LOGGER.info("Non-unique control mask for role " + i + ": " + roleControlMasks[i]);

            mRoleControlProps = null;
            break;
          }

          mRoleControlProps[i] = roleControlMasks[i].resolveIndex(roleControlMasks[i].getContents().nextSetBit(0));
        }

        if (mRoleControlProps != null)
        {
          for(int i = 0; i < mNumRoles; i++)
          {
            LOGGER.info("Role " + i + " has control prop: " + mRoleControlProps[i].sentence);
          }
        }
      }
    }

    //  If we detected that moves from multiple factors are valid in the same turn
    //  then flag the factors as requiring the inclusion of a pseudo-noop as a valid
    //  search choice every move because we'll have to choose whether to play a move
    //  from one factor or another (imposing an artificial noop on the other)
    if (mGameCharacteristics.moveChoicesFromMultipleFactors)
    {
      assert(factors != null);
      for (Factor factor : factors)
      {
        factor.setAlwaysIncludePseudoNoop(true);
      }
    }

    //	Simulate and derive a few basic stats:
    //	1) Is the game a puzzle?
    //	2) For each role what is the largest and the smallest score that seem reachable and what are the corresponding
    //     net scores
    long simulationStartTime = System.currentTimeMillis();
    //  Always do this for at least a second even if we technically don't have time to do so, since not running it
    //  at all causes all sorts of problems
    long simulationStopTime = Math.min(Math.max(timeout - (mGameCharacteristics.numRoles == 1 ? 12000 : 7000),
                                                simulationStartTime + 1000),
                                       simulationStartTime + 10000);

    double averageBranchingFactor = 0;
    double averageNumTurns = 0;
    double averageSquaredNumTurns = 0;
    double averageNumNonDrawTurns = 0;
    int numNonDrawSimulations = 0;
    int numLongDraws = 0;
    int numLongGames = 0;
    int minNumNonDrawTurns = Integer.MAX_VALUE;
    int scoreSum = -1;
    boolean isFixedSum = true;
    PlayoutInfo playoutInfo = mUnderlyingStateMachine.new PlayoutInfo(-1);

    playoutInfo.cutoffDepth = 1000;
    playoutInfo.factor = null;
    playoutInfo.moveWeights = null;

    while (System.currentTimeMillis() < simulationStopTime)
    {
      simulationsPerformed++;

      mUnderlyingStateMachine.getDepthChargeResult(initialState, playoutInfo);

      int netScore = netScore(mUnderlyingStateMachine, null);
      if (netScore != 50)
      {
        numNonDrawSimulations++;
        averageNumNonDrawTurns += playoutInfo.playoutLength;
        if (minNumNonDrawTurns > playoutInfo.playoutLength)
        {
          minNumNonDrawTurns = playoutInfo.playoutLength;
        }
      }

      int thisScoreSum = 0;

      for (int i = 0; i < mNumRoles; i++)
      {
        roleScores[i] = mUnderlyingStateMachine.getGoal(mRoleOrdering.roleIndexToRole(i));

        thisScoreSum += roleScores[i];
        if (i != 0 && mGameCharacteristics.numRoles > 2)
        {
          //	If there are several enemy players involved extract a measure
          //	of their goal correlation
          for (Role role2 : mUnderlyingStateMachine.getRoles())
          {
            if (!role2.equals(mOurRole) && !role2.equals(mRoleOrdering.roleIndexToRole(i)))
            {
              int role2Score = mUnderlyingStateMachine.getGoal(role2);

              multiRoleSamples++;
              mMultiRoleAverageScoreDiff += Math.abs(role2Score - roleScores[i]);
            }
          }
        }
      }

      if (scoreSum == -1)
      {
        scoreSum = thisScoreSum;
      }
      if (scoreSum != thisScoreSum)
      {
        isFixedSum = false;
      }

      averageNumTurns = (averageNumTurns * (simulationsPerformed - 1) + playoutInfo.playoutLength) /
                        simulationsPerformed;
      averageSquaredNumTurns = (averageSquaredNumTurns *
                                (simulationsPerformed - 1) + playoutInfo.playoutLength *
                                                             playoutInfo.playoutLength) /
                               simulationsPerformed;

      if (playoutInfo.playoutLength < minNumTurns)
      {
        assert(playoutInfo.playoutLength>0);
        minNumTurns = playoutInfo.playoutLength;
      }
      if (playoutInfo.playoutLength > maxNumTurns)
      {
        maxNumTurns = playoutInfo.playoutLength;
      }

      if (playoutInfo.playoutLength >= (maxNumTurns*95)/100)
      {
        numLongGames++;

        if (netScore == 50)
        {
          numLongDraws++;
        }
      }

      averageBranchingFactor = (averageBranchingFactor *
                                (simulationsPerformed - 1) + playoutInfo.averageBranchingFactor) /
                               simulationsPerformed;
      mGameCharacteristics.getChoicesHighWaterMark(playoutInfo.averageBranchingFactor);

      //LOGGER.debug("Saw score of ", netScore);
      if (netScore < observedMinNetScore)
      {
        observedMinNetScore = netScore;
      }

      if (netScore > observedMaxNetScore)
      {
        observedMaxNetScore = netScore;
      }
    }

    if (numNonDrawSimulations != 0)
    {
      averageNumNonDrawTurns /= numNonDrawSimulations;
    }

    LOGGER.info("branchingFactorApproximation = " + branchingFactorApproximation +
                ", averageBranchingFactor = " + averageBranchingFactor +
                ", choices high water mark = " + mGameCharacteristics.getChoicesHighWaterMark(0));
    //	Assume that a game is not really an iterated game unless it's of fixed length
    LOGGER.info("Did " + simulationsPerformed + " simulations for game characteristics");
    if ((Math.abs(branchingFactorApproximation - averageBranchingFactor) > 0.1) || (maxNumTurns != minNumTurns))
    {
      mGameCharacteristics.isIteratedGame = false;
    }

    double stdDevNumTurns = Math.sqrt(averageSquaredNumTurns -
                                      averageNumTurns * averageNumTurns);

    mGameCharacteristics.setMaxLength(maxNumTurns);
    mGameCharacteristics.setMinLength(minNumTurns);
    mGameCharacteristics.setAverageLength(averageNumTurns);
    mGameCharacteristics.setStdDeviationLength(stdDevNumTurns);
    mGameCharacteristics.setAverageNonDrawLength(averageNumNonDrawTurns);
    mGameCharacteristics.setMinNonDrawLength(minNumNonDrawTurns);
    mGameCharacteristics.setLongDrawsProportion(((double)numLongDraws)/(double)numLongGames);
    mGameCharacteristics.setAverageHyperSequenceLength(averageHyperSequenceLength);
    mGameCharacteristics.setVarianceHyperSequenceLength(varianceHyperSequenceLength);
    mGameCharacteristics.setAverageBranchingFactor(averageBranchingFactor);

    mGameCharacteristics.setEarliestCompletionDepth(mNumRoles*minNumTurns);
    if (maxNumTurns == minNumTurns)
    {
      mGameCharacteristics.setIsFixedMoveCount();
    }
    if (isFixedSum)
    {
      mGameCharacteristics.setIsFixedSum();
    }

    // Dump the game characteristics to trace output
    mGameCharacteristics.report();

    // If we're running simply for the purposes of learning game characteristics, we're all done now.
    if (mUTLearnCharacteristicsOnly)
    {
      LOGGER.info("Characteristics learned");
      return;
    }

    double avgMovesPerTurn = (double)totalMoveChoices/totalTurnSamples;
    LOGGER.info("Measured goal volatility is " +
              (totalGoalChangeCount*avgMovesPerTurn/totalTurnSamples) + (goalsMonotonic ? " [monotonic]" : " [non-monotonic]"));
    if (mGameCharacteristics.isPseudoPuzzle)
    {
      if (totalGoalChangeCount >= totalTurnSamples/(2*avgMovesPerTurn) && goalsMonotonic)
      {
        mUnderlyingStateMachine.setPlayoutPolicy(new PlayoutPolicyGoalGreedyWithPop(mUnderlyingStateMachine));
        mSearchProcessor.mUseGoalGreedy = true;
      }
      else
      {
        LOGGER.info("Not enabled Goal Greedy policy for pseudo puzzle with insufficient monotonic goal volatility");
      }
    }
    else if (!mGameCharacteristics.isSimultaneousMove &&
             mGameCharacteristics.getIsFixedSum() &&
             mGameCharacteristics.numRoles == 2)
    {
      //mUnderlyingStateMachine.setPlayoutPolicy(new PlayoutPolicyCriticalResponse(mUnderlyingStateMachine));
      //underlyingStateMachine.setPlayoutPolicy(new PlayoutPolicyLastGoodResponse(underlyingStateMachine));
    }

    heuristic.evaluateSimplicity();

    //  Don't use RAVE with heuristics that use simple application as it will swamp them
    //  Also restrict approximately to marking games (has some latched base props and has unique
    //  moves within the length of at least an average game (allows some leeway for things like Go where
    //  captures can extend the game and cause moves to be played multiple times)
    boolean useRAVE = (MachineSpecificConfiguration.getCfgBool(CfgItem.ALLOW_RAVE) &&
                       !mGameCharacteristics.isSimultaneousMove &&
                       !mGameCharacteristics.isRAVEDisabled() &&
                       mNumRoles == 2 &&
                       mUnderlyingStateMachine.getFullPropNet().getLegalPropositions().get(mOurRole).length >= mGameCharacteristics.getMaxLength() &&
                       ((mUnderlyingStateMachine.mLatches.getNumPositiveBaseLatches() > 2) ||
                        (mUnderlyingStateMachine.mLatches.getNumNegativeBaseLatches() > 2)) &&
                       (!pieceHeuristic.isEnabled() || !pieceHeuristic.applyAsSimpleHeuristic()));
    double explorationBias = 15 / (averageNumTurns + ((maxNumTurns + minNumTurns) / 2 - averageNumTurns) *
                                              stdDevNumTurns / averageNumTurns) + 0.4;
    if (explorationBias < 0.5)
    {
      explorationBias = 0.5;
    }
    else if (explorationBias > 1.2)
    {
      explorationBias = 1.2;
    }

    if (pieceHeuristic.isEnabled())
    {
      // Empirically games with piece count heuristics seem to like lower exploration bias - not entirely sure why!
      explorationBias = explorationBias * 0.7;
    }
    else if (useRAVE)
    {
      // If we guiding early expansion with RAVE exploration can be turned down somewhat.
      explorationBias *= 0.7;
    }

    mGameCharacteristics.setExplorationBias(explorationBias);
    mSearchProcessor.setExplorationBiasRange(explorationBias * 0.8, explorationBias * 1.2);

    if (mUnderlyingStateMachine.numRolloutDecisionNodeExpansions > 0)
    {
      LOGGER.info("Greedy rollout terminal discovery effectiveness: " +
                  (mUnderlyingStateMachine.greedyRolloutEffectiveness * 100) /
                  mUnderlyingStateMachine.numRolloutDecisionNodeExpansions);
      LOGGER.info("Num terminal props seen: " + mUnderlyingStateMachine.getNumTerminatingMoveProps() +
                  " out of " + mUnderlyingStateMachine.getBasePropositions().size());

      //  Greedy rollout effectiveness divided by the branching factor is a reasonable approximation
      //  of terminality density (will do for now)
      mGameCharacteristics.setTerminalityDensity(mUnderlyingStateMachine.greedyRolloutEffectiveness/(mUnderlyingStateMachine.numRolloutDecisionNodeExpansions*averageBranchingFactor));
    }

    if (simulationsPerformed > 100)
    {
      if (multiRoleSamples > 0)
      {
        mMultiRoleAverageScoreDiff /= multiRoleSamples;
      }
    }
    else
    {
      observedMinNetScore = 0;
      observedMaxNetScore = 100;
      mMultiRoleAverageScoreDiff = 0;
    }

    double greedyRolloutCost = (mUnderlyingStateMachine.numRolloutDecisionNodeExpansions == 0 ? 0
                                                                                            : averageBranchingFactor *
                                                                                              (1 - mUnderlyingStateMachine.greedyRolloutEffectiveness /
                                                                                                   (mUnderlyingStateMachine.numRolloutDecisionNodeExpansions)));

    LOGGER.info("Estimated greedy rollout cost: " + greedyRolloutCost);
    if (minNumTurns == maxNumTurns ||
        ((greedyRolloutCost > 8 ||
          stdDevNumTurns < 0.15 * averageNumTurns ||
          mUnderlyingStateMachine.greedyRolloutEffectiveness <
                                                       mUnderlyingStateMachine.numRolloutDecisionNodeExpansions / 3) &&
         (mGameCharacteristics.numRoles != 1 || mUnderlyingStateMachine.greedyRolloutEffectiveness == 0) &&
         !mUnderlyingStateMachine.mLatches.hasNegativelyLatchedGoals() &&
         !mUnderlyingStateMachine.mLatches.hasPositivelyLatchedGoals()))
    {
      if (!greedyRolloutsDisabled)
      {
        greedyRolloutsDisabled = true;
        mUnderlyingStateMachine.enableGreedyRollouts(false, true);

        //	Scale up the estimate of simulation rate since we'll be running without the overhead
        //	of greedy rollouts (which is proportional to the branching factor)
        simulationsPerformed *= (1 + greedyRolloutCost);
      }
    }

    // Attempt to solve puzzles with A* unless we've already found a solution.
    // !! Do we already know whether the game is a pseudo-puzzle at this point and could we apply A* to such a game?
    if ((!MachineSpecificConfiguration.getCfgBool(CfgItem.DISABLE_A_STAR)) &&
        (mGameCharacteristics.numRoles == 1) &&
        (observedMaxNetScore < 100) &&
        (factors == null) &&
        (mPlan == null || mPlan.isEmpty()))
    {
      tryAStar(initialState, timeout);

      if (mPlan != null && !mPlan.isEmpty())
      {
        LOGGER.info("Successfully cached plan from A* so marking as ready to play");
        return;
      }
    }

    LOGGER.info("Min raw score = " + observedMinNetScore + ", max = " + observedMaxNetScore);
    LOGGER.info("multiRoleAverageScoreDiff = " + mMultiRoleAverageScoreDiff);

    if (observedMinNetScore == observedMaxNetScore)
    {
      observedMinNetScore = 0;
      observedMaxNetScore = 100;

      LOGGER.info("No score discrimination seen during simulation - resetting to [0,100]");
    }

    if (mGameCharacteristics.numRoles == 1)
    {
      observedMinNetScore = 0;
      observedMaxNetScore = 100;

      LOGGER.info("Game is a puzzle so not normalizing scores");
    }

    //	Normalize score ranges
    mMinRawNetScore = observedMinNetScore;
    mMaxRawNetScore = observedMaxNetScore;
    mMultiRoleAverageScoreDiff = (mMultiRoleAverageScoreDiff * 100) / (mMaxRawNetScore - mMinRawNetScore);

    int rolloutSampleSize;

    if (ThreadControl.ROLLOUT_THREADS == 0)
    {
      rolloutSampleSize = 1;
    }
    else
    {
      rolloutSampleSize = (int)(simulationsPerformed / (40 * (simulationStopTime - simulationStartTime)) + 1);
      if (rolloutSampleSize > 100)
      {
        rolloutSampleSize = 100;
      }
    }

    if (useRAVE)
    {
      LOGGER.info("Use of RAVE forces sample size of 1");
      mGameCharacteristics.setRolloutSampleSize(1);
    }
    else
    {
      LOGGER.info("Not using RAVE");
      mGameCharacteristics.setRolloutSampleSize(rolloutSampleSize);
    }
    LOGGER.info("Performed " + simulationsPerformed + " simulations in " +
                (simulationStopTime - simulationStartTime) + "ms");
    LOGGER.info(simulationsPerformed *
                1000 /
                (simulationStopTime - simulationStartTime) +
                " simulations/second performed - setting rollout sample size to " +
                mGameCharacteristics.getRolloutSampleSize());

    //  If we have any time left try to optimize the way the state machine processes state changes
    if (timeout > System.currentTimeMillis() + (mGameCharacteristics.numRoles == 1 ? 11000 : 6000))
    {
      long tuningEnd = Math.min(System.currentTimeMillis() + 5000,
                                timeout - (mGameCharacteristics.numRoles == 1 ? 10000 : 5000));

      mUnderlyingStateMachine.optimizeStateTransitionMechanism(tuningEnd);
    }
    else
    {
      LOGGER.info("Insufficient time remaining for transition mechanism tuning");
    }

    if (!mGameCharacteristics.isIteratedGame || mNumRoles != 2)
    {
      if (ThreadControl.RUN_SYNCHRONOUSLY)
      {
        mGameCharacteristics.setExplorationBias(explorationBias);
      }

      mSearchProcessor.mUseRAVE = useRAVE;
      mSearchProcessor.setup(mUnderlyingStateMachine,
                            initialState,
                            mRoleOrdering,
                            mGameCharacteristics,
                            greedyRolloutsDisabled,
                            heuristic,
                            mPlan,
                            mRoleControlProps);
      mSearchProcessor.startSearch(System.currentTimeMillis() + 60000,
                                  new ForwardDeadReckonInternalMachineState(initialState),
                                  (short)0,
                                  null);

      searchUntil(timeout - SAFETY_MARGIN);
    }

    LOGGER.info("Ready to play");
  }

  private void makePreMatchAnnouncement(String xiSavedPlan)
  {
    String lGameName = getGameName();
    String lAnnouncement;
    if (xiSavedPlan != null)
    {
      lAnnouncement = "Yawn - I am playing " + lGameName;
    }
    else if (lGameName.charAt(0) >= '0' && lGameName.charAt(0) <= '9')
    {
      lAnnouncement = "Ooh - I am playing an exciting new game";
    }
    else
    {
      lAnnouncement = "I am playing " + getGameName();
    }

    lAnnouncement += " as " + mOurRole + " (Match ID " + getMatch().getMatchId() + ")";
    mBroadcaster.broadcast(lAnnouncement);
  }

  private void makePostMatchAnnouncement(boolean xiAborted)
  {
    String lAnnouncement;

    if (xiAborted)
    {
      lAnnouncement = "Match aborted.  And I was having so much fun.";
    }
    else
    {
      if (mFinalScore == 100)
      {
        if ((mGameCharacteristics.isPseudoPuzzle) || (mGameCharacteristics.getPlan() != null))
        {
          lAnnouncement = "I score 100.  That wasn't so puzzling, was it?";
        }
        else
        {
          lAnnouncement = "Perfect 100.  You'll need to try harder than that.";
        }
      }
      else if (mSolvedFromStart)
      {
        lAnnouncement = "I scored " + mFinalScore +
                                            ".  You might think that's poor, but it's the best available in this game.";
      }
      else if (mFinalScore == 0)
      {
        lAnnouncement = "Hmm, that didn't work so well.  ";
        if (mGameCharacteristics.isPseudoPuzzle)
        {
          lAnnouncement += "Looks like a need some practice.";
        }
        else
        {
          lAnnouncement += "I'll beat you next time though.";
        }
      }
      else
      {
        lAnnouncement = "I scored " + mFinalScore + ".  I don't really know what to make of that.";
      }
    }

    // If we're tidying after watchdog pop, we may not have a broadcaster.
    if (mBroadcaster != null)
    {
      mBroadcaster.broadcast(lAnnouncement);
    }
  }

  private void tryAStar(ForwardDeadReckonInternalMachineState xiInitialState, long xiTimeout)
  {
    // 8-puzzle type stuff.
    LOGGER.info("Puzzle with no observed solution");

    Set<MachineState> lGoalStates = mUnderlyingStateMachine.findGoalStates(getRole(), 90, 500, 20);
    Set<MachineState> lCleanedStates = new HashSet<>();

    for (MachineState lState : lGoalStates)
    {
      Set<GdlSentence> lEliminatedSentences = new HashSet<>();

      for (GdlSentence lSentence : lState.getContents())
      {
        int lCount = 0;

        for (MachineState lSecondState : lGoalStates)
        {
          if (lState != lSecondState &&
              unNormalizedStateDistance(lState, lSecondState) == 1 &&
              !lSecondState.getContents().contains(lSentence))
          {
            lCount++;
          }
        }

        if (lCount > 1)
        {
          lEliminatedSentences.add(lSentence);
        }
      }

      MachineState lCleaned = new MachineState(new HashSet<>(lState.getContents()));
      lCleaned.getContents().removeAll(lEliminatedSentences);

      lCleanedStates.add(lCleaned);
    }

    for(MachineState targetState : lCleanedStates)
    {
      LOGGER.info("Found possible target state: " + targetState);

      int lTargetStateSize = targetState.getContents().size();

      if (lTargetStateSize < Math.max(2, xiInitialState.size() / 2))
      {
        LOGGER.info("Unsuitable target state based on state elimination - ignoring");
      }
      else
      {
        Collection<Move> lSolution = new TargetedSolutionStatePlayer(mUnderlyingStateMachine,
                                                                     targetState,
                                                                     this).attemptAStarSolve(99,
                                                                                             xiTimeout - SAFETY_MARGIN);
        if (lSolution != null)
        {
          mPlan.considerPlan(lSolution);
          LOGGER.info("Solved by A*");
        }
        break;
      }
    }
  }

  private ForwardDeadReckonLegalMoveInfo findMoveInfo(Role role, GdlTerm moveTerm)
  {
    Move move = mUnderlyingStateMachine.getMoveFromTerm(moveTerm);
    HashSet<PolymorphicProposition> legalsForRole = new HashSet<>();
    legalsForRole.addAll(Arrays.asList(mUnderlyingStateMachine.getFullPropNet().getLegalPropositions().get(role)));
    Map<PolymorphicProposition,PolymorphicProposition> legalInputMap =
                                                            mUnderlyingStateMachine.getFullPropNet().getLegalInputMap();

    for(ForwardDeadReckonLegalMoveInfo info : mUnderlyingStateMachine.getFullPropNet().getMasterMoveList())
    {
      if (info.mMove.equals(move) && info.mInputProposition != null)
      {
        PolymorphicProposition legalProp = legalInputMap.get(info.mInputProposition);
        if (legalsForRole.contains(legalProp))
        {
          return info;
        }
      }
    }

    return null;
  }

  @Override
  public Move stateMachineSelectMove(long timeout)
    throws TransitionDefinitionException, MoveDefinitionException, GoalDefinitionException
  {
    // Kick the watchdog
    mWatchdog.alive();

    // We get the current start time
    long start = System.currentTimeMillis();
    long finishBy = timeout - SAFETY_MARGIN;
    Move bestMove;
    List<Move> moves;
    List<GdlTerm> lastJointMove = getMatch().getMostRecentMoves();

    if (lastJointMove != null)
    {
      LOGGER.info("Moves played for turn " + mTurn + ": " + lastJointMove);

      for(int lii = 0; lii < mNumRoles; lii++)
      {
        if (mPreviousTurnRootState != null && mRoleControlProps != null)
        {
          if (mPreviousTurnRootState.contains(mRoleControlProps[lii]))
          {
            mLastMoveInfo = findMoveInfo(mRoleOrdering.roleIndexToRole(lii),
                                         lastJointMove.get(mRoleOrdering.roleIndexToRawRoleIndex(lii)));
            LOGGER.info("Non-null move last turn was: " + mLastMoveInfo);
            break;
          }
        }
        else
        {
          mLastMoveInfo = findMoveInfo(mRoleOrdering.roleIndexToRole(lii),
                                       lastJointMove.get(mRoleOrdering.roleIndexToRawRoleIndex(lii)));
          if (mLastMoveInfo != null && mLastMoveInfo.mInputProposition != null)
          {
            LOGGER.info("Non-null move last turn was: " + mLastMoveInfo);
            break;
          }
        }
      }
    }

    mTurn++;
    LOGGER.info("> > > > > Starting turn " + mTurn + " < < < < <");
    StatsLogUtils.Series.TURN.logDataPoint(start, mTurn);
    mBroadcaster.broadcast("Turn " + mTurn);

    ForwardDeadReckonInternalMachineState currentState;

    mSearchProcessor.requestYield(true);

    // Check that the last move as reported by the server is the same that was sent.  Flag up if they differ because
    // there has obviously been some communication problem.
    if (mLastMove != null)
    {
      Move lLastMove = new Move(getMatch().getMostRecentMoves().get(mRoleOrdering.getOurRawRoleIndex()));
      if (!mLastMove.equals(lLastMove))
      {
        LOGGER.error("Wrong move recorded!  Submitted move was " + mLastMove + " but the server reported " + lLastMove);
      }
    }

    LOGGER.debug("Calculating current state");

    synchronized (mSearchProcessor.getSerializationObject())
    {
      mUnderlyingStateMachine.noteTurnNumber(mTurn);

      currentState = mUnderlyingStateMachine.createInternalState(getCurrentState());
      moves = mUnderlyingStateMachine.getLegalMovesCopy(currentState, mOurRole);

      if (mUnderlyingStateMachine.isTerminal(currentState))
      {
        LOGGER.warn("Asked to search in terminal state!");
        assert(false);
      }

      mPreviousTurnRootState = currentState;
    }

    if (mPlan != null && mPlan.size() > GameSearcher.thinkBelowPlanSize)
    {
      if ( mTurn == 1 && mBroadcaster != null && !isUsingConfiguredPlan() )
      {
        mBroadcaster.broadcast("Hah, too easy!  I solved it during meta-gaming!");
      }
      // We have a pre-prepared plan.  Simply play the next move.
      bestMove = mPlan.nextMove();
      LOGGER.info("Playing pre-planned move: " + bestMove);

      // We need to keep the search 'up with' the plan to make forced-play testing work properly, or else the search
      // will not be 'primed' during forced play when the plan runs out.  This is only necessary when testing with
      //  a pre-configured plan (or else plans will always go to a terminal state)
      if ( isUsingConfiguredPlan() )
      {
        mSearchProcessor.startSearch(finishBy, currentState, mCurrentMoveDepth, null);
        mCurrentMoveDepth += mNumRoles;
      }
    }
    else if (mGameCharacteristics.isIteratedGame && mNumRoles == 2)
    {
      IteratedGamePlayer iteratedPlayer = new IteratedGamePlayer(mUnderlyingStateMachine,
                                                                 this,
                                                                 mGameCharacteristics.isPseudoSimultaneousMove,
                                                                 mRoleOrdering,
                                                                 mGameCharacteristics.getCompetitivenessBonus());
      bestMove = iteratedPlayer.selectMove(moves, timeout);
      LOGGER.info("Playing best iterated game move: " + bestMove);
    }
    else
    {
      LOGGER.debug("Setting search root");
      mSearchProcessor.startSearch(finishBy, currentState, mCurrentMoveDepth, mLastMoveInfo);
      mCurrentMoveDepth += mNumRoles;

      mSearchProcessor.requestYield(false);

      LOGGER.debug("Waiting for processing");
      searchUntil(finishBy);

      LOGGER.debug("Time to submit order - ask GameSearcher to yield");
      mSearchProcessor.requestYield(true);

      long getBestMoveStartTime = System.currentTimeMillis();
      bestMove = mSearchProcessor.getBestMove();
      if (System.currentTimeMillis() - getBestMoveStartTime > 250)
      {
        LOGGER.warn("Retrieving the best move took " + (System.currentTimeMillis() - getBestMoveStartTime) + "ms");
      }

      if (!moves.contains(bestMove))
      {
        LOGGER.warn("Selected illegal move!!");
        assert(false);
        bestMove = moves.get(0);
      }
      LOGGER.info("Playing move: " + bestMove);

      mSearchProcessor.requestYield(false);
    }

    // We get the end time
    // It is mandatory that stop<timeout
    long stop = System.currentTimeMillis();

    Thread.currentThread().setPriority(Thread.NORM_PRIORITY);

    LOGGER.debug("Move took: " + (stop - start));

    if (bestMove == null)
    {
      LOGGER.error("NO MOVE FOUND!");
      System.exit(0);
    }

    mLastMove = bestMove;

    /**
     * These are functions used by other parts of the GGP codebase You
     * shouldn't worry about them, just make sure that you have moves,
     * selection, stop and start defined in the same way as this example, and
     * copy-paste these two lines in your player
     */
    notifyObservers(new GamerSelectedMoveEvent(moves, bestMove, stop - start));
    return bestMove;
  }

  private void searchUntil(long xiFinishBy)
    throws MoveDefinitionException, TransitionDefinitionException, GoalDefinitionException
  {
    try
    {
      while (System.currentTimeMillis() < xiFinishBy && !mSearchProcessor.isComplete())
      {
        if (ThreadControl.RUN_SYNCHRONOUSLY)
        {
          mSearchProcessor.expandSearch(true);
        }
        else
        {
          long sleepForUpTo = xiFinishBy - System.currentTimeMillis();

          if (sleepForUpTo > 0)
          {
            Thread.sleep(Math.min(250, sleepForUpTo));
          }
        }
      }
    }
    catch (InterruptedException lEx)
    {
      LOGGER.error("Unexpected interruption during meta-gaming", lEx);
    }

    if (mSearchProcessor.isComplete())
    {
      LOGGER.info("Early search termination because root is complete");

      if (mTurn <= 1)
      {
        // We've completed the root, either during meta-gaming or during the first turn.  As a result, this game is
        // completely solved.
        mSolvedFromStart = true;
      }
    }
  }

  @Override
  public void stateMachineStop()
  {
    // Log the final score.
    synchronized (mSearchProcessor.getSerializationObject())
    {
      ForwardDeadReckonInternalMachineState lState = mUnderlyingStateMachine.createInternalState(getCurrentState());
      mFinalScore = mUnderlyingStateMachine.getGoal(lState, mOurRole);
      LOGGER.info("Final score: " + mFinalScore);
      StatsLogUtils.Series.SCORE.logDataPoint(mFinalScore);
      StatsLogUtils.Series.SAMPLE_RATE.logDataPoint(mGameCharacteristics.getRolloutSampleSize());
    }

    // If we've just solved a puzzle for the first time, save the game history as a plan.
    if (((mFinalScore == 100) || (mSolvedFromStart)) &&
        (mGameCharacteristics.numRoles == 1) &&
        getMatch().getMoveHistory().size() > 0 &&
        (mGameCharacteristics.getPlan() == null))
    {
      mGameCharacteristics.setPlan(convertHistoryToPlan());
    }

    makePostMatchAnnouncement(false);
    tidyUp(true, false);
  }

  @Override
  public void stateMachineAbort()
  {
    LOGGER.warn("Game aborted by server");
    makePostMatchAnnouncement(true);
    tidyUp(true, false);
  }

  @Override
  public void expired()
  {
    LOGGER.warn("Game aborted on watchdog expiry");
    makePostMatchAnnouncement(true);
    tidyUp(false, false);
  }

  /**
   * Tidy up game state at the end of the game.
   *
   * @param xiRegularTermination - whether the termination is regular (stop/abort) or irregular (watchdog).
   * @param xiNewMatch - whether the start of a new match triggered this tidy-up.
   */
  private void tidyUp(boolean xiRegularTermination, boolean xiNewMatch)
  {
    StatsLogUtils.Series.TURN.logDataPoint(System.currentTimeMillis(), 999);

    // Terminate all other threads.
    if (mSearchProcessor != null)
    {
      mSearchProcessor.terminate();
      mSearchProcessor = null;
    }

    if (mSysStatsLogger != null)
    {
      mSysStatsLogger.stop();
      mSysStatsLogger = null;
    }

    if (mBroadcaster != null)
    {
      mBroadcaster.stop();
      mBroadcaster = null;
    }

    if (mWatchdog != null)
    {
      mWatchdog.stop();
      mWatchdog = null;
    }

    ThreadControl.tidyUp();

    // Save anything that we've learned about this game.
    if (xiRegularTermination)
    {
      mGameCharacteristics.saveConfig();
    }

    // Tidy up the proxy.
    if (mStateMachineProxy != null)
    {
      mStateMachineProxy.setController(null);
      mStateMachineProxy = null;
    }

    // Free off all our references.
    mOurRole                = null;
    mPlanString             = null;
    mPlan                   = null;
    mRoleOrdering           = null;
    mRoleControlProps       = null;
    mUnderlyingStateMachine = null;
    mPreviousTurnRootState  = null;
    mLogName                = null;
    mLastMove               = null;
    mLastMoveInfo           = null;

    // Reset simple variables for the next game.
    mSolvedFromStart = false;

    if (!xiNewMatch)
    {
      // Get our parent to tidy up too.
      cleanupAfterMatch();

      // Free off the static pools.
      GdlPool.drainPool();
      SymbolPool.drainPool();

      // Prompt the JVM to do garbage collection, because we've hopefully just freed a lot of stuff.
      long endGCTime = System.currentTimeMillis() + 3000;
      for (int ii = 0; ii < 1000 && System.currentTimeMillis() < endGCTime; ii++)
      {
        System.gc();
        try {Thread.sleep(1);} catch (InterruptedException lEx) {/* Whatever */}
      }
    }

    LOGGER.info("Tidy-up complete");
  }

  // Methods for use by UTs only
  public boolean utWillBeTerminal() throws TransitionDefinitionException
  {
    MachineState lState = getStateMachine().getNextState(getCurrentState(),
                                                         Collections.singletonList(mLastMove));
    return getStateMachine().isTerminal(lState);
  }

  public int utGetFinalScore()
  {
    return mFinalScore;
  }
}
