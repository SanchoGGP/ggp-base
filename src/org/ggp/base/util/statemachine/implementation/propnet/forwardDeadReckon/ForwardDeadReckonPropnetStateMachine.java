
package org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.player.gamer.statemachine.sancho.RoleOrdering;
import org.ggp.base.player.gamer.statemachine.sancho.RuntimeGameCharacteristics;
import org.ggp.base.player.gamer.statemachine.sancho.TreePath;
import org.ggp.base.util.gdl.grammar.Gdl;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.propnet.polymorphic.PolymorphicAnd;
import org.ggp.base.util.propnet.polymorphic.PolymorphicComponent;
import org.ggp.base.util.propnet.polymorphic.PolymorphicConstant;
import org.ggp.base.util.propnet.polymorphic.PolymorphicNot;
import org.ggp.base.util.propnet.polymorphic.PolymorphicOr;
import org.ggp.base.util.propnet.polymorphic.PolymorphicPropNet;
import org.ggp.base.util.propnet.polymorphic.PolymorphicProposition;
import org.ggp.base.util.propnet.polymorphic.PolymorphicTransition;
import org.ggp.base.util.propnet.polymorphic.factory.OptimizingPolymorphicPropNetFactory;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonComponent;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonComponentFactory;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState.InternalMachineStateIterator;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveSet;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropNet;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropnetFastAnimator;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonProposition;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropositionInfo;
import org.ggp.base.util.statemachine.MachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.StateMachine;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.FactorAnalyser.FactorInfo;
import org.ggp.base.util.statemachine.implementation.prover.query.ProverQueryBuilder;
import org.ggp.base.util.statemachine.playoutPolicy.IPlayoutPolicy;
import org.ggp.base.util.stats.Stats;

/**
 * A state machine.
 *
 * This class is not thread-safe.  Each instance must be accessed by a single thread.
 */
public class ForwardDeadReckonPropnetStateMachine extends StateMachine
{
  private static final Logger LOGGER = LogManager.getLogger();

  /** The underlying proposition network - in various optimised forms. */

  private final ForwardDeadReckonPropnetStateMachine                   mMaster;

  // The complete propnet.
  private ForwardDeadReckonPropNet                                     fullPropNet                     = null;

  // A propnet containing just those components required to compute the goal values when it's already known that the
  // state is terminal.
  private ForwardDeadReckonPropNet                                     goalsNet                        = null;
  private boolean                                                      useGoalNetForTerminalAndLegal   = false;
  // A propnet containing just those components required to compute the terminality of a state
  private ForwardDeadReckonPropNet                                     terminalityNet                  = null;

  // The propnet is split into two networks dependent on the proposition which changes most frequently during metagame
  // simulations.  This is commonly a "control" proposition identifying which player's turn it is in a non-simultaneous
  // game.  The X-net contains just those components required when the control proposition is true.  The O-net contains
  // just those components required when the control proposition is false.
  //
  // In games without greedy rollouts, these are set to the goalless version (below).
  private ForwardDeadReckonPropNet                                     propNetX                        = null;
  private ForwardDeadReckonPropNet                                     propNetO                        = null;

  // X- and O-nets without the goal calculation logic (which is in goalsNet).
  private ForwardDeadReckonPropNet                                     propNetXWithoutGoals            = null;
  private ForwardDeadReckonPropNet                                     propNetOWithoutGoals            = null;

  // The propnet that is currently in use (one of the X- or O- nets).
  private ForwardDeadReckonPropNet                                     propNet                         = null;
  private ForwardDeadReckonPropnetFastAnimator.InstanceInfo            propNetInstanceInfo             = null;

  private Map<Role, ForwardDeadReckonComponent[]>                      legalPropositionsX              = null;
  private Map<Role, Move[]>                                            legalPropositionMovesX          = null;
  private Map<Role, ForwardDeadReckonComponent[]>                      legalPropositionsO              = null;
  private Map<Role, Move[]>                                            legalPropositionMovesO          = null;
  private Map<Role, ForwardDeadReckonComponent[]>                      legalPropositions               = null;
  /** The player roles */
  int                                                                  numRoles;
  private Role[]                                                       roles;
  private ForwardDeadReckonInternalMachineState                        lastInternalSetStateX           = null;
  private ForwardDeadReckonInternalMachineState                        lastInternalSetStateO           = null;
  private ForwardDeadReckonInternalMachineState                        lastInternalSetState            = null;
  private final boolean                                                useSampleOfKnownLegals          = false;
  private GdlSentence                                                  XSentence                       = null;
  private ForwardDeadReckonPropositionCrossReferenceInfo               XSentenceInfo                   = null;
  private MachineState                                                 initialState                    = null;
  private ForwardDeadReckonProposition[]                               moveProps                       = null;
  private ForwardDeadReckonProposition[]                               previousMovePropsX              = null;
  private ForwardDeadReckonProposition[]                               previousMovePropsO              = null;
  private boolean                                                      measuringBasePropChanges        = false;
  private Map<ForwardDeadReckonPropositionCrossReferenceInfo, Integer> basePropChangeCounts            = new HashMap<>();
  private ForwardDeadReckonProposition[]                               chosenJointMoveProps            = null;
  private Move[]                                                       chosenMoves                     = null;
  private int[]                                                        previouslyChosenJointMovePropIdsX = null;
  private int[]                                                        previouslyChosenJointMovePropIdsO = null;
  final int[]                                                          latchedScoreRangeBuffer         = new int[2];
  private final int[]                                                  parentLatchedScoreRangeBuffer   = new int[2];
  private ForwardDeadReckonPropositionCrossReferenceInfo[]             masterInfoSet                   = null;
  private int                                                          firstBasePropIndex;
  private ForwardDeadReckonLegalMoveInfo[]                             masterLegalMoveSet              = null;
  private StateMachine                                                 validationMachine               = null;
  private RoleOrdering                                                 roleOrdering                    = null;
  private MachineState                                                 validationState                 = null;
  private int                                                          instanceId;
  private int                                                          maxInstances;
  private long                                                         metagameTimeout                 = 20000;
  private int                                                          numInstances                    = 1;
  private final Role                                                   ourRole;
  private boolean                                                      isPseudoPuzzle                  = false;
  private Set<Factor>                                                  factors                         = null;
  private StateMachineFilter                                           searchFilter                    = null;
  private ForwardDeadReckonInternalMachineState                        mNonControlMask                 = null;
  private ForwardDeadReckonInternalMachineState                        mControlMask                    = null;
  public long                                                          totalNumGatesPropagated         = 0;
  public long                                                          totalNumPropagates              = 0;
  private final Set<GdlSentence>                                       mFillerMoves                    = new HashSet<>();
  private GoalsCalculator                                              mGoalsCalculator                = null;
  private IPlayoutPolicy                                               mPlayoutPolicy                  = null;

  private final TerminalResultSet                                      mResultSet                      = new TerminalResultSet();
  // A re-usable iterator over the propositions in a machine state.
  private final InternalMachineStateIterator                           mStateIterator = new InternalMachineStateIterator();
  private final RuntimeGameCharacteristics                             mGameCharacteristics;

  //  In games with negative goal latches greedy rollouts treat state transitions that lower the opponent's
  //  maximum achievable score somewhat like transitions to winning terminal states, which is to say they
  //  preferentially select them, and preferentially avoid the converse of their own max score being
  //  reduced.  The next two parameters govern how strong that preference is (0 = pref -> 100 = always)
  //  Ideally these parameters will have natural values of 0 or 100 corresponding to the mechanism being
  //  turned on or off - anything in between implies another parameter to tune that is highly likely to be
  //  game dependent.  The most natural setting would be (100,100) which would be directly analogous with
  //  the handling of decisive win terminals, however, experimentation with ELB (the canonical game for multi-player
  //  with goal latches) shows that (100,0) works a bit better.  This feels wrong, because it is equivalent to
  //  saying (in ELB) that kings will always be captured when they can be during a rollout (fine), but nothing
  //  will be done to avoid putting a king in a position where it can be immediately captured (seems wrong).
  //  This empirical preference for (100,0) over (100,100) is something I am no entirely comfortable with, but
  //  until a counter example comes along we'll just live with (note (100,100) is still WAY better than before we had
  //  the mechanism at all, so if we had to go to that due to a counter example this is still a big step forward)
  private final int                                                    latchImprovementWeight = 100;
  private final int                                                    latchWorseningAvoidanceWeight = 0;

  //  Stack variables used during playouts that are not exposed to the caller or the policy
  private int[]                                                        playoutStackMoveInitialChoiceIndex = null;
  private int[]                                                        playoutStackMoveNextChoiceIndex = null;

  public LatchResults                                                  mLatches = new DummyLatchResults();

  /**
   * Current turn number within the overall game
   */
  volatile int                                                         mTurnNumber = 0;
  private int                                                          mLastPlayoutTurnNumber = -1;

  private class TestPropnetStateMachineStats extends Stats
  {
    private long totalResets;
    private int  numStateSettings;
    private long totalGets;
    private int  numStateFetches;
    private int  numBaseProps;
    private int  numInputs;
    private int  numLegals;

    public TestPropnetStateMachineStats(int numBaseProps,
                                        int numInputs,
                                        int numLegals)
    {
      this.numBaseProps = numBaseProps;
      this.numInputs = numInputs;
      this.numLegals = numLegals;
    }

    @Override
    public void clear()
    {
      totalResets = 0;
      numStateSettings = 0;
      totalGets = 0;
      numStateFetches = 0;
    }

    @Override
    public String toString()
    {
      StringBuilder sb = new StringBuilder();

      sb.append("#base props: " + numBaseProps);
      sb.append("\n");
      sb.append("#inputs: " + numInputs);
      sb.append("\n");
      sb.append("#legals: " + numLegals);
      sb.append("\n");
      sb.append("#state sets: " + numStateSettings);
      sb.append("\n");
      if (numStateSettings > 0)
      {
        sb.append("Average #components reset per state set: " + totalResets /
                  numStateSettings);
        sb.append("\n");
      }
      sb.append("#state gets: " + numStateFetches);
      sb.append("\n");
      if (numStateFetches > 0)
      {
        sb.append("Average #components queried per state get: " + totalGets /
                  numStateFetches);
        sb.append("\n");
      }

      return sb.toString();
    }
  }

  public class MoveWeights
  {
    public double[]  weightScore;
    private int      numSamples = 1;
    private double   total      = 0;
    private int      weightSize;
    private double[] averageScores;

    public MoveWeights(int vectorSize, int numRoles)
    {
      weightSize = vectorSize;
      weightScore = new double[vectorSize];
      averageScores = new double[numRoles];
      clear();
    }

    public void clear()
    {
      total = weightSize * 50;
      numSamples = 1;

      for (int i = 0; i < weightSize; i++)
      {
        weightScore[i] = 50;
      }
    }

    public void setWeight(int moveIndex, double weight)
    {
        weightScore[moveIndex] = weight;
    }

    public MoveWeights copy()
    {
      MoveWeights result = new MoveWeights(weightSize, averageScores.length);

      for (int i = 0; i < weightScore.length; i++)
      {
        result.weightScore[i] = weightScore[i];
      }

      result.numSamples = numSamples;
      result.total = total;

      return result;
    }

    public void addSample(double[] scores,
                          List<ForwardDeadReckonLegalMoveInfo> moves)
    {
      for (ForwardDeadReckonLegalMoveInfo move : moves)
      {
        double score = scores[move.mRoleIndex];

        double oldWeight = weightScore[move.mMasterIndex];
        double newWeigth = (oldWeight * numSamples + score) / (numSamples + 1);
        weightScore[move.mMasterIndex] = newWeigth;

        total += (newWeigth - oldWeight);
      }
      numSamples++;
    }

    public void addResult(double[] scores, ForwardDeadReckonLegalMoveInfo move)
    {
      double score = scores[move.mRoleIndex];

      double oldWeight = weightScore[move.mMasterIndex];
      double newWeigth = (oldWeight * numSamples + score) / (numSamples + 1);
      weightScore[move.mMasterIndex] = newWeigth;

      total += (newWeigth - oldWeight);
    }

    public void noteSampleComplete()
    {
      numSamples++;
    }

    public void accumulate(MoveWeights other)
    {
      total = 0;

      for (int i = 0; i < weightSize; i++)
      {
        weightScore[i] = (weightScore[i] * numSamples + other.weightScore[i] *
                                                        other.numSamples) /
                         (numSamples + other.numSamples);
        total += weightScore[i];
      }

      numSamples += other.numSamples;
    }

    public double getAverage()
    {
      return total / weightSize;
    }

    public double getStdDeviation()
    {
      double var = 0;
      double mean = total / weightSize;

      for (int i = 0; i < weightSize; i++)
      {
        var += (weightScore[i] - mean) * (weightScore[i] - mean);
      }

      return Math.sqrt(var / weightSize);
    }
  }

  /**
   * @author steve
   *  Info for a single playout (config and results)
   */
  public class PlayoutInfo
  {
    //  Config params to control the playout

    public boolean      recordTrace;
    public boolean      recordTraceStates;
    /**
     * Max depth, after which cutoff and return virtual draw
     */
    public int          cutoffDepth;
    /**
     * Move weights to apply (may be null for random)
     */
    public MoveWeights  moveWeights;
    /**
     * Factor (if any) this playout is in
     */
    public Factor       factor;

    //  Result info from playout

    /**
     * Number of moves played in this playout
     */
    public int          playoutLength;
    /**
     * Average number of move choices available during this playout
     */
    public int          averageBranchingFactor;
    /**
     * Moves played - array must be instantiated by caller and be sufficient to the max cutoff depth
     * May be null if no recorded trace is required
     */
    public final ForwardDeadReckonLegalMoveInfo[] playoutTrace;
    public final ForwardDeadReckonInternalMachineState[] statesVisited;

    public PlayoutInfo(int maxDepth)
    {
      if ( maxDepth <= 0 )
      {
        recordTrace = false;
        recordTraceStates = false;
        playoutTrace = null;
        statesVisited = null;
      }
      else
      {
        playoutTrace = new ForwardDeadReckonLegalMoveInfo[maxDepth];
        statesVisited = new ForwardDeadReckonInternalMachineState[maxDepth];
        for(int i = 0; i < maxDepth; i++)
        {
          statesVisited[i] = createEmptyInternalState();
        }
      }
    }
  }

  private TestPropnetStateMachineStats stats;

  public Stats getStats()
  {
    return stats;
  }

  public ForwardDeadReckonInternalMachineState createEmptyInternalState()
  {
    return new ForwardDeadReckonInternalMachineState(masterInfoSet, firstBasePropIndex);
  }

  public ForwardDeadReckonInternalMachineState createInternalState(MachineState state)
  {
    ForwardDeadReckonInternalMachineState result = createEmptyInternalState();

    for (GdlSentence s : state.getContents())
    {
      ForwardDeadReckonProposition p = (ForwardDeadReckonProposition)propNet.getBasePropositions().get(s);
      if ( p != null )
      {
        ForwardDeadReckonPropositionInfo info = p.getInfo();
        result.add(info);

        result.isXState |= (info.sentence == XSentence);
      }
    }

    LOGGER.trace("Created internal state: " + result + " with hash " + result.hashCode());
    return result;
  }

  public ForwardDeadReckonPropositionCrossReferenceInfo[] getInfoSet()
  {
    return masterInfoSet;
  }

  /**
   * @return the master set of legal moves - i.e. information about all moves that could ever be legal.
   */
  public ForwardDeadReckonLegalMoveInfo[] getMasterLegalMoves()
  {
    return masterLegalMoveSet;
  }

  public RoleOrdering getRoleOrdering()
  {
    return roleOrdering;
  }

  private void setRoleOrdering(RoleOrdering xiRoleOrdering)
  {
    roleOrdering = xiRoleOrdering;
  }

  /**
   * @return whether the game may be treated as a puzzle (has no
   * dependence on any other role's moves)
   */
  public boolean getIsPseudoPuzzle()
  {
    return isPseudoPuzzle;
  }

  public void performSemanticAnalysis(long timeout)
  {
    findLatches(timeout);

    if (factors == null)
    {
      PartitionedChoiceAnalyser analyzer = new PartitionedChoiceAnalyser(this);
      //  If it did not factorize does it partition? (currently we do not support both
      //  at once).  Note that this analysis must be done after the propnet is crystallized
      StateMachineFilter partitionFilter = analyzer.generatePartitionedChoiceFilter();

      if (partitionFilter != null)
      {
        setBaseFilter(partitionFilter);
      }
    }
  }

  /**
   * Find latches.
   *
   * @param xiDeadline - the (latest) time to run until.
   */
  private void findLatches(long xiDeadline)
  {
    mLatches = new LatchAnalyser(fullPropNet, this).analyse(xiDeadline, mGameCharacteristics);
    mLatches.report();
  }

  /**
   * @param xiState - state to test for latched score in
   * @return whether all roles' scores are latched.
   */
  public boolean scoresAreLatched(ForwardDeadReckonInternalMachineState xiState)
  {
    if (mGoalsCalculator != null)
    {
      if (mGoalsCalculator.scoresAreLatched(xiState))
      {
        return true;
      }
    }

    return mLatches.scoresAreLatched(xiState);
  }

  /**
   * Get the latched range of possible scores for a given role in a given state
   *
   * @param xiState - the state
   * @param xiRole - the role
   * @param xoRange - array of length 2 to contain [min,max]
   */
  public void getLatchedScoreRange(ForwardDeadReckonInternalMachineState xiState, Role xiRole, int[] xoRange)
  {
    assert(xoRange.length == 2);

    if ( mGoalsCalculator != null )
    {
      if ( mGoalsCalculator.scoresAreLatched(xiState))
      {
        xoRange[0] = mGoalsCalculator.getGoalValue(xiState, xiRole);
        xoRange[1] = xoRange[0];
        return;
      }
    }

    mLatches.getLatchedScoreRange(xiState,
                                  xiRole,
                                  fullPropNet.getGoalPropositions().get(xiRole),
                                  xoRange);
  }

  /**
   * @return instance id of this instance
   */
  public int getInstanceId()
  {
    return instanceId;
  }

  public Set<MachineState> findTerminalStates(int maxResultSet, int maxDepth)
  {
    PolymorphicProposition terminal = fullPropNet.getTerminalProposition();

    return findSupportStates(terminal.getName(), maxResultSet, maxDepth);
  }

  public Set<MachineState> findGoalStates(Role role,
                                          int minValue,
                                          int maxResultSet,
                                          int maxDepth)
  {
    Set<MachineState> results = new HashSet<>();

    for (PolymorphicProposition p : fullPropNet.getGoalPropositions().get(role))
    {
      if (Integer.parseInt(p.getName().getBody().get(1).toString()) >= minValue)
      {
        results.addAll(findSupportStates(p.getName(), maxResultSet, maxDepth));
      }
    }

    return results;
  }

  private class AntecedantCursor
  {
    //  Note that a cursor represents which (ultimately) feeding propositions of a single
    //  component need to be positive and which need to be negative to satisfy some query
    //  (which is not recorded by the cursor), and whether the component itself need be
    //  positive or negative
    public Set<PolymorphicProposition> positiveProps; //Props that must be positive for a root query to be true
    public Set<PolymorphicProposition> negativeProps; //Props that must be negative for a root query to be true
    public boolean                     isPositive;    //Whether the component this cursor is recursively expanding need
                                                      //itself be positive (else it needs to be negative)

    public AntecedantCursor()
    {
      positiveProps = new HashSet<>();
      negativeProps = new HashSet<>();
      isPositive = true;
    }

    public AntecedantCursor(AntecedantCursor parent)
    {
      positiveProps = new HashSet<>(parent.positiveProps);
      negativeProps = new HashSet<>(parent.negativeProps);
      isPositive = parent.isPositive;
    }

    public boolean compatibleWith(AntecedantCursor other)
    {
      for (PolymorphicProposition p : positiveProps)
      {
        if (other.negativeProps.contains(p))
        {
          return false;
        }
      }
      for (PolymorphicProposition p : other.positiveProps)
      {
        if (negativeProps.contains(p))
        {
          return false;
        }
      }
      for (PolymorphicProposition p : negativeProps)
      {
        if (other.positiveProps.contains(p))
        {
          return false;
        }
      }
      for (PolymorphicProposition p : other.negativeProps)
      {
        if (positiveProps.contains(p))
        {
          return false;
        }
      }

      return true;
    }

    public boolean compatibleWithAll(Set<AntecedantCursor> set)
    {
      for (AntecedantCursor c : set)
      {
        if (!compatibleWith(c))
        {
          return false;
        }
      }

      return true;
    }

    public void unionInto(AntecedantCursor c)
    {
      c.positiveProps.addAll(positiveProps);
      c.negativeProps.addAll(negativeProps);
    }

    public void unionInto(Set<AntecedantCursor> set)
    {
      if (set.isEmpty())
      {
        set.add(this);
      }
      else
      {
        for (AntecedantCursor c : set)
        {
          unionInto(c);
        }
      }
    }
  }

  public void validateStateEquality(ForwardDeadReckonPropnetStateMachine other)
  {
    if (!lastInternalSetState.equals(other.lastInternalSetState))
    {
      LOGGER.warn("Last set state mismtch");
    }

    for (PolymorphicProposition p : propNet.getBasePropositionsArray())
    {
      ForwardDeadReckonProposition fdrp = (ForwardDeadReckonProposition)p;

      if (fdrp.getValue(instanceId) != fdrp.getValue(other.instanceId))
      {
        LOGGER.warn("Base prop state mismatch on: " + p);
      }
    }
  }

  private Set<AntecedantCursor> addPropositionAntecedants(PolymorphicPropNet pn,
                                                          PolymorphicComponent p,
                                                          AntecedantCursor cursor,
                                                          int maxResultSet,
                                                          int maxDepth,
                                                          int depth)
  {
    if (depth >= maxDepth)
    {
      return null;
    }

    if (p instanceof PolymorphicTransition)
    {
      return null;
    }
    else if (p instanceof PolymorphicProposition)
    {
      PolymorphicProposition prop = (PolymorphicProposition)p;

      if (pn.getBasePropositions().values().contains(prop))
      {
        AntecedantCursor newCursor = new AntecedantCursor(cursor);

        if (cursor.isPositive)
        {
          if (!cursor.negativeProps.contains(p))
          {
            newCursor.positiveProps.add(prop);
            Set<AntecedantCursor> result = new HashSet<>();
            result.add(newCursor);
            return result;
          }
          else if (!cursor.positiveProps.contains(p))
          {
            newCursor.negativeProps.add(prop);
            Set<AntecedantCursor> result = new HashSet<>();
            result.add(newCursor);
            return result;
          }
          else
          {
            return null;
          }
        }
        if (!cursor.positiveProps.contains(p))
        {
          newCursor.negativeProps.add(prop);
          Set<AntecedantCursor> result = new HashSet<>();
          result.add(newCursor);
          return result;
        }
        else if (!cursor.negativeProps.contains(p))
        {
          newCursor.positiveProps.add(prop);
          Set<AntecedantCursor> result = new HashSet<>();
          result.add(newCursor);
          return result;
        }
        else
        {
          return null;
        }
      }

      return addPropositionAntecedants(pn,
                                       p.getSingleInput(),
                                       cursor,
                                       maxResultSet,
                                       maxDepth,
                                       depth + 1);
    }
    else if (p instanceof PolymorphicConstant)
    {
      if ( p.getValue() == cursor.isPositive )
      {
        Set<AntecedantCursor> result = new HashSet<>();
        result.add(cursor);
        return result;
      }

      return null;
    }
    else if (p instanceof PolymorphicNot)
    {
      cursor.isPositive = !cursor.isPositive;
      Set<AntecedantCursor> result = addPropositionAntecedants(pn,
                                                               p.getSingleInput(),
                                                               cursor,
                                                               maxResultSet,
                                                               maxDepth,
                                                               depth + 1);
      cursor.isPositive = !cursor.isPositive;

      return result;
    }
    else if (p instanceof PolymorphicAnd)
    {
      Set<AntecedantCursor> subResults = new HashSet<>();

      for (PolymorphicComponent c : p.getInputs())
      {
        if (subResults.size() > maxResultSet)
        {
          return null;
        }

        AntecedantCursor newCursor = new AntecedantCursor(cursor);
        Set<AntecedantCursor> inputResults = addPropositionAntecedants(pn,
                                                                       c,
                                                                       newCursor,
                                                                       maxResultSet,
                                                                       maxDepth,
                                                                       depth + 1);
        if (inputResults == null)
        {
          //	No positive matches in an AND that requires a positive result => failure
          if (cursor.isPositive)
          {
            return null;
          }
        }
        else
        {
          if (cursor.isPositive)
          {
            //	We require ALL inputs, so take the conditions gathered for this one and validate
            //	consistency with the current cursor, then add them into that condition set
            if (subResults.isEmpty())
            {
              subResults = inputResults;
            }
            else
            {
              Set<AntecedantCursor> validInputResults = new HashSet<>();

              for (AntecedantCursor cur : inputResults)
              {
                for (AntecedantCursor subResult : subResults)
                {
                  if (subResult.compatibleWith(cur))
                  {
                    AntecedantCursor combinedResult = new AntecedantCursor(subResult);
                    cur.unionInto(combinedResult);

                    validInputResults.add(combinedResult);
                  }
                }
              }

              subResults = validInputResults;
            }
          }
          else
          {
            //	This is a OR when viewed in the negative sense, so we just need one, and each such
            //	match is a new results set
            subResults.addAll(inputResults);
          }
        }
      }

      return subResults;
    }
    else if (p instanceof PolymorphicOr)
    {
      Set<AntecedantCursor> subResults = new HashSet<>();

      for (PolymorphicComponent c : p.getInputs())
      {
        if (subResults.size() > maxResultSet)
        {
          return null;
        }

        AntecedantCursor newCursor = new AntecedantCursor(cursor);
        Set<AntecedantCursor> inputResults = addPropositionAntecedants(pn,
                                                                       c,
                                                                       newCursor,
                                                                       maxResultSet,
                                                                       maxDepth,
                                                                       depth + 1);
        if (inputResults == null)
        {
          //	Any positive matches in an OR that requires a negative result => failure
          if (!cursor.isPositive)
          {
            return null;
          }
        }
        else
        {
          if (!cursor.isPositive)
          {
            //	We require ALL inputs to be negative, so take the conditions gathered for this one and validate
            //	consistency with the current cursor, then add them into that condition set
            if (subResults.isEmpty())
            {
              subResults = inputResults;
            }
            else
            {
              Set<AntecedantCursor> validInputResults = new HashSet<>();

              for (AntecedantCursor cur : inputResults)
              {
                for (AntecedantCursor subResult : subResults)
                {
                  if (subResult.compatibleWith(cur))
                  {
                    AntecedantCursor combinedResult = new AntecedantCursor(subResult);
                    cur.unionInto(combinedResult);

                    validInputResults.add(combinedResult);
                  }
                }
              }

              subResults = validInputResults;
            }
          }
          else
          {
            //	Any positive will do, and each such
            //	match is a new results set
            subResults.addAll(inputResults);
          }
        }
      }

      return subResults;
    }

    throw new RuntimeException("Unknown component");
  }

  public Set<MachineState> findSupportStates(GdlSentence queryProposition,
                                             int maxResultSet,
                                             int maxDepth)
  {
    Set<MachineState> result = new HashSet<>();

    PolymorphicProposition p = fullPropNet.findProposition(queryProposition);
    if (p != null)
    {
      Set<AntecedantCursor> cursorSet = addPropositionAntecedants(fullPropNet,
                                                                  p,
                                                                  new AntecedantCursor(),
                                                                  maxResultSet,
                                                                  maxDepth,
                                                                  0);

      if ( cursorSet != null )
      {
        for (AntecedantCursor c : cursorSet)
        {
          //  We currently only look for states that have positive requirements (some set
          //  or true propositions is enough to satisfy regardless of state of other
          //  propositions).  This could be readily expanded to a required plus a required-not
          //  set as the AntecedantCursor contains this information
          if ( c.negativeProps.isEmpty() )
          {
            MachineState satisfyingState = new MachineState(new HashSet<GdlSentence>());

            for (PolymorphicProposition prop : c.positiveProps)
            {
              satisfyingState.getContents().add(prop.getName());
            }

            result.add(satisfyingState);
          }
        }
      }
    }

    return result;
  }

  /**
   * UT-only state machine constructor.
   */
  public ForwardDeadReckonPropnetStateMachine()
  {
    maxInstances = 1;
    ourRole = null;
    mGameCharacteristics = null;
    mMaster = this;
  }

  /**
   * Construct the master state machine, for use during meta-gaming and for cloning per-thread instances from.
   *
   * State machines are NOT thread-safe.
   *
   * @param xiMaxInstances         - the maximum number of clones that will be created.
   * @param xiMetaGameTimeout      - meta-gaming timeout.
   * @param xiOurRole              - our role.
   * @param xiGameCharacteristics  - the game characteristics (read-write).
   */
  public ForwardDeadReckonPropnetStateMachine(int xiMaxInstances,
                                              long xiMetaGameTimeout,
                                              Role xiOurRole,
                                              RuntimeGameCharacteristics xiGameCharacteristics)
  {
    maxInstances = xiMaxInstances;
    metagameTimeout = xiMetaGameTimeout;
    ourRole = xiOurRole;
    mGameCharacteristics = xiGameCharacteristics;
    mMaster = this;
  }

  /**
   * Private constructor for creating per-thread state machine instances.  Called by createInstance() on the master.
   *
   * @param master     - the master state machine to clone.
   * @param instanceId - the ID of this instance.
   */
  private ForwardDeadReckonPropnetStateMachine(ForwardDeadReckonPropnetStateMachine master, int instanceId)
  {
    mMaster = master;
    maxInstances = -1;
    this.instanceId = instanceId;
    propNetX = master.propNetX;
    propNetO = master.propNetO;
    propNetXWithoutGoals = master.propNetXWithoutGoals;
    propNetOWithoutGoals = master.propNetOWithoutGoals;
    enableGreedyRollouts = master.enableGreedyRollouts;
    goalsNet = master.goalsNet;
    terminalityNet = master.terminalityNet;
    XSentence = master.XSentence;
    XSentenceInfo = master.XSentenceInfo;
    legalPropositionMovesX = master.legalPropositionMovesX;
    legalPropositionMovesO = master.legalPropositionMovesO;
    legalPropositionsX = master.legalPropositionsX;
    legalPropositionsO = master.legalPropositionsO;
    legalPropositions = master.legalPropositions;
    initialState = master.initialState;
    firstBasePropIndex = master.firstBasePropIndex;
    roles = master.roles;
    numRoles = master.numRoles;
    fullPropNet = master.fullPropNet;
    masterInfoSet = master.masterInfoSet;
    factors = master.factors;
    mLatches = master.mLatches;
    ourRole = master.ourRole;
    setRoleOrdering(master.getRoleOrdering());
    totalNumMoves = master.totalNumMoves;
    if ( master.mGoalsCalculator != null )
    {
      mGoalsCalculator = master.mGoalsCalculator.createThreadSafeReference();
    }
    mGameCharacteristics = master.mGameCharacteristics;
    mControlMask = master.mControlMask;
    mNonControlMask = master.mNonControlMask;
    removeOldBasePropsBeforeAddingNew = master.removeOldBasePropsBeforeAddingNew;
    use2passBasePropSet = master.use2passBasePropSet;
    mPlayoutPolicy = (master.mPlayoutPolicy == null ? null : master.mPlayoutPolicy.cloneFor(this));

    stateBufferX1 = createEmptyInternalState();
    stateBufferX2 = createEmptyInternalState();
    stateBufferO1 = createEmptyInternalState();
    stateBufferO2 = createEmptyInternalState();
    maskStateBuffer = createEmptyInternalState();

    for(int i = 0; i < rolloutDecisionStack.length; i++)
    {
      rolloutDecisionStack[i] = new RolloutDecisionState();
    }

    moveProps = new ForwardDeadReckonProposition[numRoles];
    previousMovePropsX = new ForwardDeadReckonProposition[numRoles];
    previousMovePropsO = new ForwardDeadReckonProposition[numRoles];
    chosenJointMoveProps = new ForwardDeadReckonProposition[numRoles];
    chosenMoves = new Move[numRoles];
    previouslyChosenJointMovePropIdsX = new int[numRoles];
    previouslyChosenJointMovePropIdsO = new int[numRoles];
    isPseudoPuzzle = master.isPseudoPuzzle;

    stats = new TestPropnetStateMachineStats(fullPropNet.getBasePropositions().size(),
                                             fullPropNet.getInputPropositions().size(),
                                             fullPropNet.getLegalPropositions().get(getRoles()[0]).length);
  }

  /**
   * @return a new state machine instance (for use by a new thread).
   */
  public ForwardDeadReckonPropnetStateMachine createInstance()
  {
    if (numInstances >= maxInstances)
    {
      throw new RuntimeException("Too many instances");
    }

    ForwardDeadReckonPropnetStateMachine result = new ForwardDeadReckonPropnetStateMachine(this, numInstances++);

    return result;
  }

  @Override
  public void initialize(List<Gdl> description)
  {
    // Log the GDL so that we can play again as required.
    StringBuffer lGDLString  = new StringBuffer();
    lGDLString.append("GDL\n");
    for (Gdl element : description)
    {
      lGDLString.append(element);
      lGDLString.append('\n');
    }
    LOGGER.debug(lGDLString.toString());

    setRandomSeed(1);

    try
    {
      fullPropNet = (ForwardDeadReckonPropNet)OptimizingPolymorphicPropNetFactory.create(
                                                                               description,
                                                                               new ForwardDeadReckonComponentFactory());
      fullPropNet.renderToFile("propnet_001.dot");
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.removeAnonymousPropositions(fullPropNet);
      fullPropNet.renderToFile("propnet_012_AnonRemoved.dot");
      LOGGER.debug("Num components after anon prop removal: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.removeUnreachableBasesAndInputs(fullPropNet);
      fullPropNet.renderToFile("propnet_014_UnreachablesRemoved.dot");
      LOGGER.debug("Num components after unreachable removal: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      isPseudoPuzzle = OptimizingPolymorphicPropNetFactory.removeIrrelevantBasesAndInputs(fullPropNet,
                                                                                          ourRole,
                                                                                          mFillerMoves);
      fullPropNet.renderToFile("propnet_016_IrrelevantRemoved.dot");
      LOGGER.debug("Num components after irrelevant removal: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.removeRedundantConstantsAndGates(fullPropNet, false);
      fullPropNet.renderToFile("propnet_018_RedundantRemoved.dot");
      LOGGER.debug("Num components after first pass redundant components removal: " +
                   fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.refactorLargeGates(fullPropNet);
      fullPropNet.renderToFile("propnet_020_BeforeLargeFanout.dot");
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.refactorLargeFanouts(fullPropNet);
      fullPropNet.renderToFile("propnet_030_AfterLargeFanout.dot");
      LOGGER.debug("Num components after large gate refactoring: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.removeDuplicateLogic(fullPropNet);
      LOGGER.debug("Num components after duplicate removal: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.optimizeInputSets(fullPropNet);
      LOGGER.debug("Num components after input set optimization: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.optimizeInvertedInputs(fullPropNet);
      LOGGER.debug("Num components after inverted input optimization: " + fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      OptimizingPolymorphicPropNetFactory.removeRedundantConstantsAndGates(fullPropNet, true);
      LOGGER.debug("Num components after further removal of redundant components: " +
                   fullPropNet.getComponents().size());
      assert(fullPropNet.validateClosure());

      // Ensure that no propositions apart from strict input props (base, does, init) have any outputs, as this is
      // assumed by the fast animator.  Accordingly we re-wire slightly such that if any such do exist we replace their
      // output connection by one from their input (which they anyway just directly forward, so this also removes a
      // small propagation step).
      OptimizingPolymorphicPropNetFactory.removeNonBaseOrDoesPropositionOutputs(fullPropNet);
      assert(fullPropNet.validateClosure());

      fullPropNet.renderToFile("propnet_040_Reduced.dot");
      roles = fullPropNet.getRoles();
      numRoles = roles.length;
      roleOrdering = new RoleOrdering(this, ourRole);
      setRoleOrdering(roleOrdering);

      moveProps = new ForwardDeadReckonProposition[numRoles];
      previousMovePropsX = new ForwardDeadReckonProposition[numRoles];
      previousMovePropsO = new ForwardDeadReckonProposition[numRoles];
      chosenJointMoveProps = new ForwardDeadReckonProposition[numRoles];
      chosenMoves = new Move[numRoles];
      previouslyChosenJointMovePropIdsX = new int[numRoles];
      previouslyChosenJointMovePropIdsO = new int[numRoles];
      stats = new TestPropnetStateMachineStats(fullPropNet.getBasePropositions().size(),
                                               fullPropNet.getInputPropositions().size(),
                                               fullPropNet.getLegalPropositions().get(getRoles()[0]).length);
      //	Assess network statistics
      int numInputs = 0;
      int numMultiInputs = 0;
      int numMultiInputComponents = 0;

      for (PolymorphicComponent c : fullPropNet.getComponents())
      {
        int n = c.getInputs().size();

        numInputs += n;

        if (n > 1)
        {
          numMultiInputComponents++;
          numMultiInputs += n;
        }
      }

      int numComponents = fullPropNet.getComponents().size();
      LOGGER.debug("Num components: " + numComponents + " with an average of " + numInputs / numComponents + " inputs.");
      LOGGER.debug("Num multi-input components: " +
                   numMultiInputComponents +
                   " with an average of " +
                   (numMultiInputComponents == 0 ? "N/A" : numMultiInputs /
                                                           numMultiInputComponents) +
                   " inputs.");

      int numGoals = 0;
      for(PolymorphicProposition[] goals : fullPropNet.getGoalPropositions().values())
      {
        numGoals += goals.length;
      }
      assert(numGoals>0);
      masterInfoSet = new ForwardDeadReckonPropositionCrossReferenceInfo[fullPropNet.getBasePropositions().size() + numGoals + 1];
      int index = 0;

      // Ensure the goal propositions always appear in the same order (for saving/reloading of latches, etc.).  For ease
      // of debugging, these are ordered first by role (in GDL order) and then by increasing goal value.
      for(PolymorphicProposition lGoalProp : fullPropNet.getOrderedGoalPropositions())
      {
        ForwardDeadReckonPropositionCrossReferenceInfo info = new ForwardDeadReckonPropositionCrossReferenceInfo();

        info.sentence = lGoalProp.getName();
        info.fullNetProp = (ForwardDeadReckonProposition)lGoalProp;
        info.xNetProp = (ForwardDeadReckonProposition)lGoalProp;
        info.oNetProp = (ForwardDeadReckonProposition)lGoalProp;
        info.goalsNetProp = (ForwardDeadReckonProposition)lGoalProp;
        info.index = index;

        masterInfoSet[index++] = info;

        ((ForwardDeadReckonProposition)lGoalProp).setInfo(info);
      }

      {
        ForwardDeadReckonProposition prop = (ForwardDeadReckonProposition)fullPropNet.getTerminalProposition();
        ForwardDeadReckonPropositionCrossReferenceInfo info = new ForwardDeadReckonPropositionCrossReferenceInfo();

        info.sentence = prop.getName();
        info.fullNetProp = prop;
        info.xNetProp = prop;
        info.oNetProp = prop;
        info.goalsNetProp = prop;
        info.terminalityNetProp = prop;
        info.index = index;

        masterInfoSet[index++] = info;

        prop.setInfo(info);
      }

      assert(index == numGoals+1);
      firstBasePropIndex = index;

      // Ensure the base propositions always appear in the same order across multiple runs.  This is required for
      // neural network reloading.
      List<GdlSentence> lKeys = new ArrayList<>(fullPropNet.getBasePropositions().keySet());
      Collections.sort(lKeys, new Comparator<GdlSentence>()
      {
        @Override
        public int compare(GdlSentence aa, GdlSentence bb)
        {
          return aa.toString().compareTo(bb.toString());
        }
      });

      for (GdlSentence lKey : lKeys)
      {
        ForwardDeadReckonProposition prop = (ForwardDeadReckonProposition)fullPropNet.getBasePropositions().get(lKey);
        ForwardDeadReckonPropositionCrossReferenceInfo info = new ForwardDeadReckonPropositionCrossReferenceInfo();

        info.sentence = lKey;
        info.fullNetProp = prop;
        info.xNetProp = prop;
        info.oNetProp = prop;
        info.goalsNetProp = prop;
        info.terminalityNetProp = prop;
        info.index = index;

        masterInfoSet[index++] = info;

        prop.setInfo(info);
        basePropChangeCounts.put(info, 0);
      }

      fullPropNet.crystalize(masterInfoSet, firstBasePropIndex, null, maxInstances);
      masterLegalMoveSet = fullPropNet.getMasterMoveList();

      // Try to factor the game.  But...
      // - Don't do it if we know there's only 1 factor.
      // - Allow no more than half the remaining time.
      // - Don't do it if we've previously timed out whilst factoring this game and we don't have at least 25% more time
      //   now.
      long factorizationAnalysisTimeout = (metagameTimeout - System.currentTimeMillis()) / 2;

      if (mGameCharacteristics != null)
      {
        FactorAnalyser factorAnalyser = new FactorAnalyser(this);
        FactorInfo lFactorInfo = factorAnalyser.run(factorizationAnalysisTimeout, mGameCharacteristics);

        factors = lFactorInfo.mFactors;
        if (factors != null)
        {
          LOGGER.info("Game factorizes into " + factors.size() + " factors");
        }

        mControlMask = createEmptyInternalState();

        if (lFactorInfo.mControlSet != null)
        {
          for (PolymorphicProposition p : lFactorInfo.mControlSet)
          {
            ForwardDeadReckonPropositionInfo info = ((ForwardDeadReckonProposition)p).getInfo();
            mControlMask.add(info);
          }
        }
        mNonControlMask = new ForwardDeadReckonInternalMachineState(mControlMask);
        mNonControlMask.invert();
      }

      for (ForwardDeadReckonPropositionInfo info : masterInfoSet)
      {
        ForwardDeadReckonPropositionCrossReferenceInfo crInfo = (ForwardDeadReckonPropositionCrossReferenceInfo)info;

        crInfo.xNetPropId = crInfo.xNetProp.id;
        crInfo.oNetPropId = crInfo.oNetProp.id;
      }

      stateBufferX1 = createEmptyInternalState();
      stateBufferX2 = createEmptyInternalState();
      stateBufferO1 = createEmptyInternalState();
      stateBufferO2 = createEmptyInternalState();
      maskStateBuffer = createEmptyInternalState();

      for(int i = 0; i < rolloutDecisionStack.length; i++)
      {
        rolloutDecisionStack[i] = new RolloutDecisionState();
      }

      fullPropNet.reset(false);
      ForwardDeadReckonProposition initProp = (ForwardDeadReckonProposition)fullPropNet.getInitProposition();
      if ( initProp != null && initProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
      {
        fullPropNet.animator.getInstanceInfo(0).changeComponentValueTo(initProp.id, true);
      }
      propNet = fullPropNet;
      propNetInstanceInfo = propNet.animator.getInstanceInfo(0);
      initialState = getInternalStateFromBase(createEmptyInternalState()).getMachineState();
      fullPropNet.reset(true);

      measuringBasePropChanges = true;

      try
      {
        for (int i = 0; i < 10; i++)
        {
          performDepthCharge(initialState, null);
        }
      }
      catch (TransitionDefinitionException | MoveDefinitionException lEx)
      {
        LOGGER.warn("Exception performing depth charges", lEx);
      }
      measuringBasePropChanges = false;

      int highestCount = 0;
      for (Entry<ForwardDeadReckonPropositionCrossReferenceInfo, Integer> e : basePropChangeCounts.entrySet())
      {
        if (e.getValue() > highestCount)
        {
          highestCount = e.getValue();
          XSentence = e.getKey().sentence;
        }
      }

      basePropChangeCounts = null;
      lastInternalSetState = null;
      lastGoalState = null;
      propNet = null;

      for(int i = 0; i < previousMovePropsO.length; i++)
      {
        previousMovePropsO[i] = null;
      }

      propNetX = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
      propNetO = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
      goalsNet = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
      terminalityNet = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
      propNetX.removeInits();
      propNetO.removeInits();
      assert(propNetX.validateClosure());

      if (XSentence != null)
      {
        GdlSentence OSentence = null;

        LOGGER.info("Reducing with respect to XSentence: " + XSentence);
        OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetX, XSentence, true);
        assert(propNetX.validateClosure());

        //	If the reduced net always transitions it's own hard-wired sentence into the opposite state
        //	it may be part of a pivot whereby control passes between alternating propositions.  Check this
        //	Do we turn something else on unconditionally?
        for (Entry<GdlSentence, PolymorphicProposition> e : propNetX.getBasePropositions().entrySet())
        {
          PolymorphicComponent input = e.getValue().getSingleInput();

          if (input instanceof PolymorphicTransition)
          {
            PolymorphicComponent driver = input.getSingleInput();

            if (driver instanceof PolymorphicConstant && driver.getValue())
            {
              //	Found a suitable candidate
              OSentence = e.getKey();
              break;
            }
          }
        }

        if (OSentence != null)
        {
          LOGGER.debug("Possible OSentence: " + OSentence);
          OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetO, OSentence, true);

          //	Does this one turn the original back on?
          PolymorphicProposition originalPropInSecondNet = propNetO.getBasePropositions().get(XSentence);
          if (originalPropInSecondNet != null)
          {
            PolymorphicComponent input = originalPropInSecondNet.getSingleInput();

            if (input instanceof PolymorphicTransition)
            {
              PolymorphicComponent driver = input.getSingleInput();

              if (!(driver instanceof PolymorphicConstant) || !driver.getValue())
              {
                //	Nope - doesn't work
                OSentence = null;
                LOGGER.debug("Fails to recover back-transition to " + XSentence);
              }
            }
          }

          if (OSentence != null)
          {
            //	So if we set the first net's trigger condition to off in the second net do we find
            //	the second net's own trigger is always off?
            OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetO, XSentence, false);

            PolymorphicProposition OSentenceInSecondNet = propNetO.getBasePropositions().get(OSentence);
            if (OSentenceInSecondNet != null)
            {
              PolymorphicComponent input = OSentenceInSecondNet.getSingleInput();

              if (input instanceof PolymorphicTransition)
              {
                PolymorphicComponent driver = input.getSingleInput();

                if (!(driver instanceof PolymorphicConstant) || driver.getValue())
                {
                  //	Nope - doesn't work
                  LOGGER.info("Fails to recover back-transition remove of " + OSentence);
                  OSentence = null;
                }

                //	Finally, if we set the OSentence off in the first net do we recover the fact that
                //	the XSentence always moves to off in transitions from the first net?
                if (OSentence != null)
                {
                  OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetX, OSentence, false);
                  assert(propNetX.validateClosure());

                  PolymorphicProposition XSentenceInFirstNet = propNetX.getBasePropositions().get(XSentence);
                  if (XSentenceInFirstNet != null)
                  {
                    input = XSentenceInFirstNet.getSingleInput();

                    if (input instanceof PolymorphicTransition)
                    {
                      driver = input.getSingleInput();

                      if (!(driver instanceof PolymorphicConstant) ||
                          driver.getValue())
                      {
                        //	Nope - doesn't work
                        LOGGER.debug("Fails to recover removal of " + XSentence);
                        OSentence = null;
                      }
                    }
                  }
                }
              }
            }
            else
            {
              OSentence = null;
            }
          }
        }

        if (OSentence == null)
        {
          LOGGER.debug("Reverting OSentence optimizations");
          //	Failed - best we can do is simply drive the XSentence to true in one network
          propNetX = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
          propNetO = new ForwardDeadReckonPropNet(fullPropNet, new ForwardDeadReckonComponentFactory());
          propNetX.removeInits();
          propNetO.removeInits();
          OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetX, XSentence, true);
          OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetO, XSentence, false);
        }
        //OptimizingPolymorphicPropNetFactory.fixBaseProposition(propNetO, XSentence, false);
        propNetX.renderToFile("propnet_050_ReducedX.dot");
        propNetO.renderToFile("propnet_060_ReducedO.dot");
        LOGGER.debug("Num components remaining in X-net: " + propNetX.getComponents().size());
        LOGGER.debug("Num components remaining in O-net: " + propNetO.getComponents().size());
      }

      propNetXWithoutGoals = new ForwardDeadReckonPropNet(propNetX, new ForwardDeadReckonComponentFactory());
      propNetOWithoutGoals = new ForwardDeadReckonPropNet(propNetO, new ForwardDeadReckonComponentFactory());
      propNetXWithoutGoals.removeGoals();
      propNetOWithoutGoals.removeGoals();
      OptimizingPolymorphicPropNetFactory.minimizeNetwork(propNetXWithoutGoals);
      OptimizingPolymorphicPropNetFactory.minimizeNetwork(propNetOWithoutGoals);
      propNetXWithoutGoals.renderToFile("propnet_070_XWithoutGoals.dot");
      propNetOWithoutGoals.renderToFile("propnet_080_OWithoutGoals.dot");

      terminalityNet.removeAllButTerminal();
      goalsNet.removeAllButGoals();

      goalsNet.renderToFile("propnet_090_GoalsReduced.dot");

      LOGGER.info("Num components in goal-less X-net: " + propNetXWithoutGoals.getComponents().size());
      LOGGER.info("Num components in goal-less O-net: " + propNetOWithoutGoals.getComponents().size());
      LOGGER.info("Num components in goal net:        " + goalsNet.getComponents().size());
      LOGGER.info("Num components in terminality net: " + terminalityNet.getComponents().size());

      //masterInfoSet = new ForwardDeadReckonPropositionCrossReferenceInfo[fullPropNet
      //    .getBasePropositions().size()];
      //index = 0;
      finalizePropositionCrossReferenceInfo();

      propNetX.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);
      propNetO.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);
      goalsNet.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);
      terminalityNet.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);

      for(ForwardDeadReckonPropositionInfo info : masterInfoSet)
      {
        ForwardDeadReckonPropositionCrossReferenceInfo crInfo = (ForwardDeadReckonPropositionCrossReferenceInfo)info;

        crInfo.xNetPropId = crInfo.xNetProp.id;
        crInfo.oNetPropId = crInfo.oNetProp.id;
      }

      terminalityNet.reset(true);
      goalsNet.reset(true);
      //	Force calculation of the goal set while we're single threaded
      goalsNet.getGoalPropositions();

      //  Set move factor info
      if ( factors != null )
      {
        //  Moves with no dependencies (typically a noop) can appear in multiple factors, but
        //  should be tagged as factor-free
        setMoveInfoForPropnet(propNetX);
        setMoveInfoForPropnet(propNetO);
      }

//      stateBufferX1 = new ForwardDeadReckonInternalMachineState(masterInfoSet);
//      stateBufferX2 = new ForwardDeadReckonInternalMachineState(masterInfoSet);
//      stateBufferO1 = new ForwardDeadReckonInternalMachineState(masterInfoSet);
//      stateBufferO2 = new ForwardDeadReckonInternalMachineState(masterInfoSet);

      propNetX.reset(true);
      propNetO.reset(true);
      //	Force calculation of the goal set while we're single threaded
      propNetX.getGoalPropositions();
      propNetO.getGoalPropositions();

      propNet = propNetX;
      propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
      legalPropositions = legalPropositionsX;

      totalNumMoves = fullPropNet.getMasterMoveList().length;

    }
    catch (InterruptedException e)
    {
      // TODO: handle exception
    }

    PolymorphicPropNet.sLastSourceToTargetMap = null;
  }

  private void finalizePropositionCrossReferenceInfo()
  {
    //  Cross-reference the base propositions of the various networks
    for (Entry<GdlSentence, PolymorphicProposition> e : fullPropNet.getBasePropositions().entrySet())
    {
      ForwardDeadReckonProposition oProp = (ForwardDeadReckonProposition)propNetO
          .getBasePropositions().get(e.getKey());
      ForwardDeadReckonProposition xProp = (ForwardDeadReckonProposition)propNetX
          .getBasePropositions().get(e.getKey());
      ForwardDeadReckonProposition goalsProp = (ForwardDeadReckonProposition)goalsNet
          .getBasePropositions().get(e.getKey());
      ForwardDeadReckonProposition terminalityProp = (ForwardDeadReckonProposition)terminalityNet
          .getBasePropositions().get(e.getKey());
      ForwardDeadReckonProposition fullNetPropFdr = (ForwardDeadReckonProposition)e.getValue();
      ForwardDeadReckonPropositionCrossReferenceInfo info = (ForwardDeadReckonPropositionCrossReferenceInfo)fullNetPropFdr.getInfo();

      info.xNetProp = xProp;
      info.oNetProp = oProp;
      info.goalsNetProp = goalsProp;
      info.terminalityNetProp = terminalityProp;

      xProp.setInfo(info);
      oProp.setInfo(info);
      if (goalsProp != null)
      {
        goalsProp.setInfo(info);
      }
      if (terminalityProp != null)
      {
        terminalityProp.setInfo(info);
      }

      if (e.getKey().equals(XSentence))
      {
        XSentenceInfo = info;
      }
    }

    //  Cross-reference the goal propositions of the various networks
    for(Entry<Role, PolymorphicProposition[]> e : fullPropNet.getGoalPropositions().entrySet())
    {
      for(PolymorphicProposition prop : e.getValue())
      {
        ForwardDeadReckonPropositionInfo info = ((ForwardDeadReckonProposition)prop).getInfo();
        if ( info != null )
        {
          for(PolymorphicProposition p : propNetO.getGoalPropositions().get(e.getKey()))
          {
            if ( p.getName() == prop.getName() )
            {
              ((ForwardDeadReckonPropositionCrossReferenceInfo)info).oNetProp = (ForwardDeadReckonProposition)p;
              ((ForwardDeadReckonProposition)p).setInfo(info);
              break;
            }
          }

          for(PolymorphicProposition p : propNetX.getGoalPropositions().get(e.getKey()))
          {
            if ( p.getName() == prop.getName() )
            {
              ((ForwardDeadReckonPropositionCrossReferenceInfo)info).xNetProp = (ForwardDeadReckonProposition)p;
              ((ForwardDeadReckonProposition)p).setInfo(info);
              break;
            }
          }

          for(PolymorphicProposition p : goalsNet.getGoalPropositions().get(e.getKey()))
          {
            if ( p.getName() == prop.getName() )
            {
              ((ForwardDeadReckonPropositionCrossReferenceInfo)info).goalsNetProp = (ForwardDeadReckonProposition)p;
              ((ForwardDeadReckonProposition)p).setInfo(info);
              break;
            }
          }
        }
      }
    }

    //  Cross-reference the terminal propositions of the various networks
    {
      ForwardDeadReckonProposition prop = (ForwardDeadReckonProposition)fullPropNet.getTerminalProposition();
      ForwardDeadReckonProposition oProp = (ForwardDeadReckonProposition)propNetO.getTerminalProposition();
      ForwardDeadReckonProposition xProp = (ForwardDeadReckonProposition)propNetX.getTerminalProposition();
      ForwardDeadReckonProposition goalsProp = (ForwardDeadReckonProposition)goalsNet.getTerminalProposition();
      ForwardDeadReckonProposition terminalityProp = (ForwardDeadReckonProposition)terminalityNet.getTerminalProposition();
      ForwardDeadReckonPropositionCrossReferenceInfo info = (ForwardDeadReckonPropositionCrossReferenceInfo)prop.getInfo();

      assert(info != null);

      info.oNetProp = oProp;
      info.xNetProp = xProp;
      info.goalsNetProp = goalsProp;
      info.terminalityNetProp = terminalityProp;

      oProp.setInfo(info);
      xProp.setInfo(info);
      goalsProp.setInfo(info);
      terminalityProp.setInfo(info);
    }
  }

  public void optimizeStateTransitionMechanism(long timeout)
  {
    ForwardDeadReckonInternalMachineState initialInternalState = createInternalState(initialState);
    Role firstRole = getRoles()[0];
    int withTrueCount = 0;
    int withFalseCount = 0;
    long totalTime = (timeout - System.currentTimeMillis());

    //  Perform some measurements to see if the state machine runs faster if we reset
    //  base props removed between states first or add ones set first (this varies
    //  significantly between games).  We budget 2 seconds (1 second of simulation for each choice)
    //  at the end of state machine initialization for this.  The random seed is set to the same
    //  value at the start of each set of simulations to ensure the same games are played, and thus
    //  the same state transitions occur
    removeOldBasePropsBeforeAddingNew = false;

    PlayoutInfo playoutInfo = new PlayoutInfo(-1);
    playoutInfo.cutoffDepth = 1000;

    setRandomSeed(100);
    long startWithFalse = System.currentTimeMillis();
    while(System.currentTimeMillis() < startWithFalse + totalTime/2)
    {
      getDepthChargeResult(initialInternalState, playoutInfo);
      withFalseCount++;
    }

    removeOldBasePropsBeforeAddingNew = true;

    setRandomSeed(100);
    long startWithTrue = System.currentTimeMillis();
    while(System.currentTimeMillis() < startWithTrue + totalTime/2)
    {
      getDepthChargeResult(initialInternalState, playoutInfo);
      withTrueCount++;
    }

    LOGGER.info("Iterations in " + totalTime/2 + " ms with/without removal before adding: " + withTrueCount + "/" + withFalseCount);
    if ( (Math.abs(withTrueCount-withFalseCount)*100)/Math.max(withTrueCount, withFalseCount) < 4 )
    {
      use2passBasePropSet = false;
      LOGGER.info("Speed improvement insufficient to justify 2-pass prop setting");
    }
    else
    {
      use2passBasePropSet = true;
      if ( withFalseCount > withTrueCount )
      {
        removeOldBasePropsBeforeAddingNew = false;
        LOGGER.info("Speed improvement of " + (100*(withFalseCount-withTrueCount))/withFalseCount + "% adding new base props before removing old");
      }
      else
      {
        LOGGER.info("Speed improvement of " + (100*(withTrueCount-withFalseCount))/withTrueCount + "% removing old base props before adding new");
      }
    }
  }

  private void setMoveInfoForPropnet(ForwardDeadReckonPropNet pn)
  {
    //  Moves with no dependencies (typically a noop) can appear in multiple factors, but
    //  should be tagged as factor-free
    Set<ForwardDeadReckonLegalMoveInfo> multiFactorMoves = new HashSet<>();

    ForwardDeadReckonLegalMoveInfo[] factoredIndexMoveList = fullPropNet.getMasterMoveList();
    for(ForwardDeadReckonLegalMoveInfo info : pn.getMasterMoveList())
    {
      if ( info != null )
      {
        if ( factors != null )
        {
          for(Factor factor : factors)
          {
            if ( factor.getMoveInfos().contains(factoredIndexMoveList[info.mMasterIndex]))
            {
              if ( info.mFactor != null )
              {
                multiFactorMoves.add(info);
              }
              info.mFactor = factor;
            }
          }
        }

        if ( info.mInputProposition != null && mFillerMoves.contains(info.mInputProposition.getName()))
        {
          info.mIsVirtualNoOp = true;
        }
      }
    }

    if ( factors != null )
    {
      for(ForwardDeadReckonLegalMoveInfo info : multiFactorMoves)
      {
        info.mFactor = null;
      }
    }
  }

  /**
   * @return the underlying full propnet.
   */
  public ForwardDeadReckonPropNet getFullPropNet()
  {
    return fullPropNet;
  }

  /**
   * @return the underlying goal propnet.
   */
  public ForwardDeadReckonPropNet getGoalPropNet()
  {
    return goalsNet;
  }

  /**
   * @return the underlying goal-less X-net.
   */
  public ForwardDeadReckonPropNet getXPropNet()
  {
    return propNetXWithoutGoals;
  }

  /**
   * @return the underlying goal-less O-net.
   */
  public ForwardDeadReckonPropNet getOPropNet()
  {
    return propNetOWithoutGoals;
  }

  /**
   * Return a search filter for use with this state machine when performing higher level goal search.
   *
   * @return filter to use
   */
  public StateMachineFilter getBaseFilter()
  {
    if (searchFilter == null)
    {
      searchFilter = new NullStateMachineFilter();
    }

    return searchFilter;
  }

  /**
   * Note a new turn has started of the specified number (within the game)
   * @param turn
   */
  public void noteTurnNumber(int turn)
  {
    mTurnNumber = turn;
  }

  private void setBaseFilter(StateMachineFilter filter)
  {
    searchFilter = filter;
  }

  /**
   * Get a state mask for the non-control propositions
   * @return null if unknown else state mask
   */
  public ForwardDeadReckonInternalMachineState getNonControlMask()
  {
    return mNonControlMask;
  }

  /**
   * @return number of roles in the game
   */
  public int getNumRoles()
  {
    return numRoles;
  }

  /**
   * Get a state mask for the control propositions
   * @return null if unknown else state mask
   */
  public ForwardDeadReckonInternalMachineState getControlMask()
  {
    return mControlMask;
  }

  private void setBasePropositionsFromState(MachineState state)
  {
    setBasePropositionsFromState(createInternalState(state));
  }

  private ForwardDeadReckonInternalMachineState stateBufferX1 = null;
  private ForwardDeadReckonInternalMachineState stateBufferX2 = null;
  private ForwardDeadReckonInternalMachineState stateBufferO1 = null;
  private ForwardDeadReckonInternalMachineState stateBufferO2 = null;

  private boolean use2passBasePropSet               = true;
  private boolean removeOldBasePropsBeforeAddingNew = true;

  private void makeBasePropChangesMeasured(ForwardDeadReckonInternalMachineState nextInternalSetState)
  {
    InternalMachineStateIterator lIterator = mStateIterator;

    lIterator.reset(lastInternalSetState);
    while (lIterator.hasNext())
    {
      ForwardDeadReckonPropositionInfo info = lIterator.next();
      ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
      int propId = (propNet == propNetX ? infoCr.xNetPropId : infoCr.oNetPropId);
      if ( propId != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
      {
        if (nextInternalSetState.contains(info))
        {
          propNetInstanceInfo.changeComponentValueTo(propId, true);
        }
        else
        {
          propNetInstanceInfo.changeComponentValueTo(propId, false);
        }
      }

      basePropChangeCounts.put(infoCr,
                               basePropChangeCounts.get(infoCr) + 1);
    }
  }

  private void makeBasePropChangesUnmeasured(ForwardDeadReckonInternalMachineState nextInternalSetState)
  {
    //  The following code is a bit baroque, in the interests of extracting the last possible bit of performance, as
    //  this routine is a significant hotspot in games with large state.  The gist is:
    //  1) Iteration uses low level methods for more direct access
    //  2) Set are done before resets, or visa-versa (depending on statistical checks made during metagaming)
    //  3) Two minimize the overhead of state bitmap enumeration the first pass notes the range ends for the second
    //     pass
    //  4) The first pass is split into two sections to minimize the execution of logic specific to identifying the
    //     second pass range endpoints
    if ( propNet == propNetX)
    {
      assert(propNetInstanceInfo == propNetX.animator.getInstanceInfo(instanceId));

      if ( !use2passBasePropSet )
      {
        int index = lastInternalSetState.contents.nextSetBit(lastInternalSetState.firstBasePropIndex);

        while(index != -1)
        {
          ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
          if ( infoCr.xNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
          {
            propNetInstanceInfo.changeComponentValueTo(infoCr.xNetProp.id, nextInternalSetState.contents.fastGet(index));
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }
      }
      else
      {
        int index = lastInternalSetState.contents.nextSetBit(0);
        int firstIndex = -1;
        int lastIndex = -1;

        while(index != -1)
        {
          boolean needsSetting = nextInternalSetState.contents.fastGet(index);
          if ( removeOldBasePropsBeforeAddingNew != needsSetting )
          {
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
            if ( infoCr.xNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
            {
              propNetInstanceInfo.changeComponentValueTo(infoCr.xNetProp.id, needsSetting);
            }
          }
          else
          {
            firstIndex = index;
            lastIndex = index;
            index = lastInternalSetState.contents.nextSetBit(index+1);
            break;
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }
        while(index != -1)
        {
          boolean needsSetting = nextInternalSetState.contents.fastGet(index);
          if ( removeOldBasePropsBeforeAddingNew != needsSetting )
          {
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
            if ( infoCr.xNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
            {
              propNetInstanceInfo.changeComponentValueTo(infoCr.xNetProp.id, needsSetting);
            }
          }
          else
          {
            lastIndex = index;
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }

        if ( firstIndex != -1 )
        {
          do
          {
            if ( removeOldBasePropsBeforeAddingNew == nextInternalSetState.contents.fastGet(firstIndex) )
            {
              ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[firstIndex];
              if ( infoCr.xNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
              {
                propNetInstanceInfo.changeComponentValueTo(infoCr.xNetProp.id, removeOldBasePropsBeforeAddingNew);
              }
            }

            if ( firstIndex == lastIndex )
            {
              break;
            }
            firstIndex = lastInternalSetState.contents.nextSetBit(firstIndex+1);
          } while(true);
        }
      }
    }
    else
    {
      assert(propNetInstanceInfo == propNetO.animator.getInstanceInfo(instanceId));

      if ( !use2passBasePropSet )
      {
        int index = lastInternalSetState.contents.nextSetBit(lastInternalSetState.firstBasePropIndex);

        while(index != -1)
        {
          ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
          if ( infoCr.oNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
          {
            propNetInstanceInfo.changeComponentValueTo(infoCr.oNetProp.id, nextInternalSetState.contents.fastGet(index));
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }
      }
      else
      {
        int index = lastInternalSetState.contents.nextSetBit(0);
        int firstIndex = -1;
        int lastIndex = -1;

        while(index != -1)
        {
          boolean needsSetting = nextInternalSetState.contents.fastGet(index);
          if ( removeOldBasePropsBeforeAddingNew != needsSetting )
          {
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
            if ( infoCr.oNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
            {
              propNetInstanceInfo.changeComponentValueTo(infoCr.oNetProp.id, needsSetting);
            }
          }
          else
          {
            firstIndex = index;
            lastIndex = index;
            index = lastInternalSetState.contents.nextSetBit(index+1);
            break;
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }
        while(index != -1)
        {
          boolean needsSetting = nextInternalSetState.contents.fastGet(index);
          if ( removeOldBasePropsBeforeAddingNew != nextInternalSetState.contents.fastGet(index) )
          {
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[index];
            if ( infoCr.oNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
            {
              propNetInstanceInfo.changeComponentValueTo(infoCr.oNetProp.id, needsSetting);
            }
          }
          else
          {
            lastIndex = index;
          }

          index = lastInternalSetState.contents.nextSetBit(index+1);
        }

        if ( firstIndex != -1 )
        {
          do
          {
            if ( removeOldBasePropsBeforeAddingNew == nextInternalSetState.contents.fastGet(firstIndex) )
            {
              ForwardDeadReckonPropositionCrossReferenceInfo infoCr = masterInfoSet[firstIndex];
              if ( infoCr.oNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
              {
                propNetInstanceInfo.changeComponentValueTo(infoCr.oNetProp.id, removeOldBasePropsBeforeAddingNew);
              }
            }

            if ( firstIndex == lastIndex )
            {
              break;
            }
            firstIndex = lastInternalSetState.contents.nextSetBit(firstIndex+1);
          } while(true);
        }
      }
    }
  }

  private void makeBasePropChangesWithReset()
  {
    InternalMachineStateIterator lIterator = mStateIterator;

    lIterator.reset(lastInternalSetState);
    while (lIterator.hasNext())
    {
      ForwardDeadReckonPropositionInfo s = lIterator.next();
      ForwardDeadReckonPropositionCrossReferenceInfo sCr = (ForwardDeadReckonPropositionCrossReferenceInfo)s;
      int compId = (propNet == propNetX ? sCr.xNetProp.id : sCr.oNetProp.id);

      if ( compId != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId )
      {
        propNetInstanceInfo.changeComponentValueTo(compId, true);
      }
    }
  }

  private void setBasePropositionsFromState(ForwardDeadReckonInternalMachineState state)
  {
    if (lastInternalSetState != null)
    {
      if (!lastInternalSetState.equals(state))
      {
        ForwardDeadReckonInternalMachineState nextInternalSetState;

        lastInternalSetState.xor(state);
        if (propNet == propNetX)
        {
          nextInternalSetState = (lastInternalSetState == stateBufferX1 ? stateBufferX2 : stateBufferX1);
        }
        else
        {
          nextInternalSetState = (lastInternalSetState == stateBufferO1 ? stateBufferO2 : stateBufferO1);
        }
        assert(nextInternalSetState != state);
        nextInternalSetState.copy(state);

        if (!measuringBasePropChanges)
        {
          makeBasePropChangesUnmeasured(nextInternalSetState);
        }
        else
        {
          makeBasePropChangesMeasured(nextInternalSetState);
        }

        lastInternalSetState = nextInternalSetState;
      }
    }
    else
    {
      lastInternalSetState = new ForwardDeadReckonInternalMachineState(state);

      makeBasePropChangesWithReset();
    }
  }

  /**
   * Computes if the state is terminal. Should return the value of the terminal
   * proposition for the state.
   */
  @Override
  public boolean isTerminal(MachineState state)
  {
    ForwardDeadReckonInternalMachineState internalState = createInternalState(state);
    return isTerminal(internalState);
  }

  public boolean isTerminal(ForwardDeadReckonInternalMachineState state)
  {
    setPropNetUsage(state);
    setBasePropositionsFromState(state);

    return isTerminal();
  }

  public boolean isTerminalDedicated(ForwardDeadReckonInternalMachineState state)
  {
    if ( factors != null )
    {
      return isTerminal(state);
    }

    setTerminalityNetBasePropsFromState(state);

    return terminalityNet.getActiveBaseProps(instanceId).contains(((ForwardDeadReckonProposition)fullPropNet.getTerminalProposition()).getInfo());
  }

  public boolean isTerminal()
  {
    if ( factors != null && !hasAvailableMoveForAllRoles(propNet) )
    {
      return true;
    }

    return isTerminalUnfactored();
  }

  private boolean isTerminalUnfactored()
  {
    boolean result = propNet.getActiveBaseProps(instanceId).contains(((ForwardDeadReckonProposition)fullPropNet.getTerminalProposition()).getInfo());

    if (validationMachine != null)
    {
      if (validationMachine.isTerminal(validationState) != result)
      {
        LOGGER.warn("Terminality mismatch");
      }
    }
    return result;
  }

  /**
   * Computes the goal for a role in the current state. Should return the value
   * of the goal proposition that is true for that role. If there is not
   * exactly one goal proposition true for that role, then you should throw a
   * GoalDefinitionException because the goal is ill-defined.
   */
  @Override
  public int getGoal(MachineState state, Role role)
  {
    ForwardDeadReckonInternalMachineState internalState = createInternalState(state);
    return getGoal(internalState, role);
  }

  public int getGoal(Role role)
  {
    return getGoal((ForwardDeadReckonInternalMachineState)null, role);
  }

  /**
   * Returns the initial state. The initial state can be computed by only
   * setting the truth value of the INIT proposition to true, and then
   * computing the resulting state.
   */
  @Override
  public MachineState getInitialState()
  {
    LOGGER.trace("Initial state: " + initialState);
    return initialState;
  }

  /**
   * Computes the legal moves for role in state.
   */
  @Override
  public List<Move> getLegalMoves(MachineState state, Role role)
  {
    ForwardDeadReckonInternalMachineState internalState = createInternalState(state);

    return getLegalMovesCopy(internalState, role);
  }

  public List<Move> getLegalMovesCopy(ForwardDeadReckonInternalMachineState state, Role role)
  {
    List<Move> result;

    ForwardDeadReckonLegalMoveSet moveSet = getLegalMoveSet(state);

    result = new LinkedList<>();
    for (ForwardDeadReckonLegalMoveInfo moveInfo : moveSet.getContents(role))
    {
      result.add(moveInfo.mMove);
    }

    return result;
  }

  /**
   * @return the legal moves in the specified state for the specified role.
   *
   * @param state - the state.
   * @param role  - the role.
   *
   * WARNING: This version of the function returns a collection backed by a pre-allocated array.  It is only suitable
   *          for immediate use and not to be stored.
   */
  public Collection<ForwardDeadReckonLegalMoveInfo> getLegalMoves(ForwardDeadReckonInternalMachineState state,
                                                                  Role role)
  {
    Collection<ForwardDeadReckonLegalMoveInfo> lResult = getLegalMoveSet(state).getContents(role);
    assert(lResult.size() > 0);
    return lResult;
  }

  /**
   * @return the legal moves in the specified state as a ForwardDeadReckonLegalMoveSet.
   *
   * @param state - the state.
    *
   * WARNING: This version of the function returns a collection backed by a pre-allocated array.  It is only suitable
   *          for immediate use and not to be stored.
   */
  public ForwardDeadReckonLegalMoveSet getLegalMoveSet(ForwardDeadReckonInternalMachineState state)
  {
    setPropNetUsage(state);
    setBasePropositionsFromState(state);
    return propNet.getActiveLegalProps(instanceId);
  }

  /**
   * @return whether a specified move is legal for a role in a state.
   *
   * @param state - the state.
   * @param role  - the role.
   * @param move  - the proposed move.
   *
   * @throws MoveDefinitionException if the GDL is malformed.
   */
  public boolean isLegalMove(MachineState state, Role role, Move move) throws MoveDefinitionException
  {
    setPropNetUsage(state);
    setBasePropositionsFromState(state);

    Map<GdlSentence, PolymorphicProposition> inputProps = propNet.getInputPropositions();

    GdlSentence moveSentence = ProverQueryBuilder.toDoes(role, move);
    PolymorphicProposition moveInputProposition = inputProps.get(moveSentence);
    PolymorphicProposition legalProp = propNet.getLegalInputMap().get(moveInputProposition);
    if (legalProp != null)
    {
      return ((ForwardDeadReckonComponent)legalProp.getSingleInput()).getValue(instanceId);
    }

    throw new MoveDefinitionException(state, role);
  }

  private void setPropNetUsage(MachineState state)
  {
    setPropNetUsage(createInternalState(state));
  }

  private void setPropNetUsage(ForwardDeadReckonInternalMachineState state)
  {
    if (XSentence != null)
    {
      if (state.isXState)
      {
        if (propNet != propNetX)
        {
          propNet = propNetX;
          propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
          legalPropositions = legalPropositionsX;

          lastInternalSetStateO = lastInternalSetState;
          lastInternalSetState = lastInternalSetStateX;
        }
      }
      else
      {
        if (propNet != propNetO)
        {
          propNet = propNetO;
          propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
          legalPropositions = legalPropositionsO;

          lastInternalSetStateX = lastInternalSetState;
          lastInternalSetState = lastInternalSetStateO;
        }
      }
    }
  }

  /**
   * Computes the next state given state and the list of moves.
   */
  @Override
  public MachineState getNextState(MachineState state, List<Move> moves)
  {
    //RuntimeOptimizedComponent.getCount = 0;
    //RuntimeOptimizedComponent.dirtyCount = 0;
    ForwardDeadReckonInternalMachineState internalState = createInternalState(state);

    setPropNetUsage(internalState);

    ForwardDeadReckonInternalMachineState internalResult = createEmptyInternalState();
    ForwardDeadReckonLegalMoveInfo[] internalMoves = new ForwardDeadReckonLegalMoveInfo[moves.size()];

    Map<GdlSentence, PolymorphicProposition> inputProps = propNet.getInputPropositions();
    Map<PolymorphicProposition, PolymorphicProposition> legalInputMap = propNet.getLegalInputMap();
    int moveRawIndex = 0;

    for (GdlSentence moveSentence : toDoes(moves))
    {
      ForwardDeadReckonProposition moveInputProposition = (ForwardDeadReckonProposition)inputProps.get(moveSentence);
      ForwardDeadReckonLegalMoveInfo moveInfo;

      if (moveInputProposition != null)
      {
        ForwardDeadReckonProposition legalProp = (ForwardDeadReckonProposition)legalInputMap.get(moveInputProposition);

        moveInfo = propNet.getMasterMoveList()[legalProp.getInfo().index];
      }
      else
      {
        moveInfo = new ForwardDeadReckonLegalMoveInfo();

        moveInfo.mIsPseudoNoOp = true;
      }

      int internalMoveIndex = (roleOrdering == null ? moveRawIndex : roleOrdering.rawRoleIndexToRoleIndex(moveRawIndex));
      internalMoves[internalMoveIndex] = moveInfo;
      moveRawIndex++;
    }

    getNextState(internalState, null, internalMoves, internalResult);

    MachineState result = getInternalStateFromBase(createEmptyInternalState()).getMachineState();

    return result;
  }

  /**
   * Get the next state given the current state and a set of moves.  Write the resulting state directly into the
   * supplied new state buffer.
   *
   * @param state       - the original state.
   * @param factor      - the factor.
   * @param moves       - the moves to make from the original state - in INTERNAL ordering
   * @param xbNewState  - the buffer into which the new state is written.
   */
  public void getNextState(ForwardDeadReckonInternalMachineState state,
                           Factor factor,
                           ForwardDeadReckonLegalMoveInfo[] moves,
                           ForwardDeadReckonInternalMachineState xbNewState)
  {
    assert(xbNewState != null);

    setPropNetUsage(state);

    int movesCount = 0;
    int nonNullMovesCount = 0;

    for (ForwardDeadReckonLegalMoveInfo move : moves)
    {
      ForwardDeadReckonProposition moveProp = move.mIsPseudoNoOp ? null : move.mInputProposition;
      moveProps[movesCount++] = moveProp;
      if ( moveProp != null )
      {
        nonNullMovesCount++;
      }
    }
    setBasePropositionsFromState(state);

    for (int i = 0; i < movesCount; i++)
    {
      ForwardDeadReckonProposition moveProp =  moveProps[i];
      ForwardDeadReckonProposition previousMoveProp = (propNet == propNetX ? previousMovePropsX[i] : previousMovePropsO[i]);

      if ( previousMoveProp != moveProp )
      {
        if ( propNet == propNetX )
        {
          previousMovePropsX[i] = moveProps[i];
        }
        else
        {
          previousMovePropsO[i] = moveProps[i];
        }

        if ( moveProp != null )
        {
          propNetInstanceInfo.changeComponentValueTo(moveProp.id, true);
        }
        if ( previousMoveProp != null  )
        {
          propNetInstanceInfo.changeComponentValueTo(previousMoveProp.id, false);
        }
      }
    }

    propNet.getActiveBaseProps(instanceId).markDirty();
    getInternalStateFromBase(xbNewState);

    if ( nonNullMovesCount == 0 )
    {
      ForwardDeadReckonInternalMachineState nonControlMask;
      ForwardDeadReckonInternalMachineState controlMask;

      if ( factor != null )
      {
        nonControlMask = factor.getStateMask(true);
        controlMask = factor.getInverseStateMask(true);
      }
      else
      {
        nonControlMask = getNonControlMask();
        controlMask = getControlMask();
      }

      if ( controlMask != null )
      {
        //  Hack - re-impose the base props from the starting state.  We need to do it this
        //  way in order for the non-factor turn logic (control prop, step, etc) to generate
        //  correctly, but then make sure we have not changed any factor-specific base props
        //  which can happen because no moves were played (consider distinct clauses on moves)
        ForwardDeadReckonInternalMachineState basePropState = new ForwardDeadReckonInternalMachineState(state);

        basePropState.intersect(nonControlMask);
        xbNewState.intersect(controlMask);
        xbNewState.merge(basePropState);
      }
    }
  }

  private boolean transitionToNextStateFromChosenMove()
  {
    if (validationMachine != null)
    {
      List<Move> moves = new LinkedList<>();

      for (Move move : chosenMoves)
      {
        moves.add(move);
      }
      try
      {
        validationMachine.getNextState(validationState, moves);
      }
      catch (TransitionDefinitionException e)
      {
        e.printStackTrace();
      }
    }

    int index = 0;
    for (ForwardDeadReckonProposition moveProp : chosenJointMoveProps)
    {
      int previousChosenMoveId;

      if (propNet == propNetX)
      {
        previousChosenMoveId = previouslyChosenJointMovePropIdsX[index];
      }
      else
      {
        previousChosenMoveId = previouslyChosenJointMovePropIdsO[index];
      }

      int movePropId = -1;
      if (moveProp != null)
      {
        movePropId = moveProp.id;
      }

      if (previousChosenMoveId != movePropId)
      {
        if ( movePropId != -1 )
        {
          propNetInstanceInfo.changeComponentValueTo(movePropId, true);
        }
        if ( previousChosenMoveId != -1 )
        {
          propNetInstanceInfo.changeComponentValueTo(previousChosenMoveId, false);
        }
      }
      if (propNet == propNetX)
      {
        previouslyChosenJointMovePropIdsX[index++] = movePropId;
      }
      else
      {
        previouslyChosenJointMovePropIdsO[index++] = movePropId;
      }
    }

    propagateCalculatedNextState();

    return true;
  }

  /* Already implemented for you */
  @Override
  public Role[] getRoles()
  {
    return roles;
  }

  /* Helper methods */

  /**
   * The Input propositions are indexed by (does ?player ?action). This
   * translates a list of Moves (backed by a sentence that is simply ?action)
   * into GdlSentences that can be used to get Propositions from
   * inputPropositions. and accordingly set their values etc. This is a naive
   * implementation when coupled with setting input values, feel free to change
   * this for a more efficient implementation.
   *
   * @param moves
   * @return
   */
  private List<GdlSentence> toDoes(Move[] moves)
  {
    List<GdlSentence> doeses = new ArrayList<>(moves.length);
    Map<Role, Integer> roleIndices = getRoleIndices();

    for (Role lRole : roles)
    {
      int index = roleIndices.get(lRole);
      doeses.add(ProverQueryBuilder.toDoes(lRole, moves[index]));
    }
    return doeses;
  }

  /**
   * The Input propositions are indexed by (does ?player ?action). This
   * translates a list of Moves (backed by a sentence that is simply ?action)
   * into GdlSentences that can be used to get Propositions from
   * inputPropositions. and accordingly set their values etc. This is a naive
   * implementation when coupled with setting input values, feel free to change
   * this for a more efficient implementation.
   *
   * @param moves
   * @return
   */
  private List<GdlSentence> toDoes(List<Move> moves)
  {
    List<GdlSentence> doeses = new ArrayList<>(moves.size());
    Map<Role, Integer> roleIndices = getRoleIndices();

    for (Role lRole : roles)
    {
      int index = roleIndices.get(lRole);
      doeses.add(ProverQueryBuilder.toDoes(lRole, moves.get(index)));
    }
    return doeses;
  }

  private void propagateCalculatedNextState()
  {
    ForwardDeadReckonInternalMachineState transitionTo = propNet.getActiveBaseProps(instanceId);

    boolean targetIsXNet = transitionTo.contains(XSentenceInfo);
    if (propNet == propNetX)
    {
      if (!targetIsXNet)
      {
        propNet = propNetO;
        propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
        lastInternalSetStateX = lastInternalSetState;
        lastInternalSetState = lastInternalSetStateO;

        legalPropositions = legalPropositionsO;
      }
    }
    else
    {
      if (targetIsXNet)
      {
        propNet = propNetX;
        propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
        lastInternalSetStateO = lastInternalSetState;
        lastInternalSetState = lastInternalSetStateX;

        legalPropositions = legalPropositionsX;
      }
    }

    transitionTo.isXState = targetIsXNet;

    setBasePropositionsFromState(transitionTo);
  }

  private ForwardDeadReckonInternalMachineState getInternalStateFromBase(ForwardDeadReckonInternalMachineState xbState)
  {
    assert(xbState != propNet.getActiveBaseProps(instanceId));

    xbState.copy(propNet.getActiveBaseProps(instanceId));
    xbState.isXState = (XSentenceInfo != null && xbState.contains(XSentenceInfo));

    return xbState;
  }


  private Map<Role, List<Move>> recentLegalMoveSetsList = new HashMap<>();

  @Override
  public Move getRandomMove(MachineState state, Role role)
      throws MoveDefinitionException
  {
    if (useSampleOfKnownLegals)
    {
      int choiceSeed = getRandom(100);
      final int tryPreviousPercentage = 80;
      List<Move> previouslyAvailableMoves = null;
      boolean preferNew = false;

      if (choiceSeed < tryPreviousPercentage &&
          recentLegalMoveSetsList.keySet().contains(role))
      {
        previouslyAvailableMoves = recentLegalMoveSetsList.get(role);
        Move result = previouslyAvailableMoves.get(getRandom(previouslyAvailableMoves.size()));

        if (isLegalMove(state, role, result))
        {
          return result;
        }
      }
      else if (choiceSeed > 100 - tryPreviousPercentage / 2)
      {
        preferNew = true;
      }

      List<Move> legals = getLegalMoves(state, role);
      List<Move> candidates;

      if (preferNew && previouslyAvailableMoves != null)
      {
        candidates = new LinkedList<>();

        for (Move move : legals)
        {
          if (!previouslyAvailableMoves.contains(move))
          {
            candidates.add(move);
          }
        }
      }
      else
      {
        candidates = legals;
      }

      if (legals.size() > 1)
      {
        recentLegalMoveSetsList.put(role, legals);
      }

      return candidates.get(getRandom(candidates.size()));
    }
    List<Move> legals = getLegalMoves(state, role);

    int randIndex = getRandom(legals.size());
    return legals.get(randIndex);
  }

  private class RolloutDecisionState
  {
    public RolloutDecisionState()
    {
      // TODO Auto-generated constructor stub
    }

    //  The following arrays may be null or point to a pre-allocated buffers
    public ForwardDeadReckonLegalMoveInfo[] chooserMoves;
    public boolean[]                        propProcessed;
    public int                              numChoices;
    //  The following are pre-allocated buffers used repeatedly to avoid GC
    //  It is expanded as necessary but never shrunk
    private ForwardDeadReckonLegalMoveInfo[] chooserMovesBuffer;
    private boolean[]                       propProcessedBuffer;
    public final ForwardDeadReckonProposition[]   nonChooserProps = new ForwardDeadReckonProposition[numRoles];
    public int                              chooserIndex;
    public int                              baseChoiceIndex;
    public int                              nextChoiceIndex;
    public int                              rolloutSeq;
    public int                              maxAchievableOpponentScoreTotal;
    final ForwardDeadReckonInternalMachineState   state = createEmptyInternalState();
    Role                                    choosingRole;

    void  clearMoveChoices()
    {
      numChoices = -1;
      chooserMoves = null;
      propProcessed = null;
    }

    void  setNumMoveChoices(int numMoveChoices)
    {
      if ( chooserMovesBuffer == null || chooserMovesBuffer.length < numMoveChoices )
      {
        //  Allow extra so we don't have to repeatedly expand
        chooserMovesBuffer = new ForwardDeadReckonLegalMoveInfo[numMoveChoices*2];
        propProcessedBuffer = new boolean[numMoveChoices*2];
      }

      numChoices = numMoveChoices;
      chooserMoves = chooserMovesBuffer;
      propProcessed = propProcessedBuffer;
    }
  }

  private final RolloutDecisionState[] rolloutDecisionStack = new RolloutDecisionState[TreePath.MAX_PATH_LEN];
  private int                    rolloutStackDepth;
  private int                    rolloutSeq           = 0;

  private int                    totalRoleoutChoices;
  private int                    totalRoleoutNodesExamined;

  private void doRecursiveGreedyRoleout(TerminalResultSet results,
                                        Factor factor,
                                        MoveWeights moveWeights,
                                        ForwardDeadReckonLegalMoveInfo[] playedMoves,
                                        int cutoffDepth)
  {
    Move hintMove = null;
    int hintMoveDepth = -1;

    do
    {
      Move winningMove = transitionToNextStateInGreedyRollout(results,
                                                              factor,
                                                              hintMove,
                                                              moveWeights,
                                                              playedMoves,
                                                              rolloutStackDepth);
      if (winningMove != null)
      {
        hintMove = winningMove;
        hintMoveDepth = rolloutStackDepth;

        //	Next player had a 1 move forced win.  Pop the stack and choose again at this level unless deciding player was
        //	the same as for this node
        //	TODO - this doesn't handle well games in which the same player gets to play multiple times
        if (rolloutStackDepth > 0 &&
            rolloutDecisionStack[rolloutStackDepth - 1].nextChoiceIndex != rolloutDecisionStack[rolloutStackDepth - 1].baseChoiceIndex)
        {
          if (rolloutDecisionStack[rolloutStackDepth].chooserIndex != rolloutDecisionStack[rolloutStackDepth - 1].chooserIndex)
          {
            rolloutDecisionStack[rolloutStackDepth].chooserMoves = null;

            RolloutDecisionState poppedState = rolloutDecisionStack[--rolloutStackDepth];

            setPropNetUsage(poppedState.state);
            setBasePropositionsFromState(poppedState.state);
          }
          else
          {
            if (!isTerminal() && !scoresAreLatched(lastInternalSetState))
            {
              if (rolloutStackDepth++ >= hintMoveDepth)
              {
                hintMove = null;
              }
            }
            else
            {
              results.considerResult(rolloutDecisionStack[rolloutStackDepth++].choosingRole);
              break;
            }
          }
        }
        else
        {
          if (!isTerminal() && !scoresAreLatched(lastInternalSetState))
          {
            if (rolloutStackDepth++ >= hintMoveDepth)
            {
              hintMove = null;
            }
          }
          else
          {
            results.considerResult(rolloutDecisionStack[rolloutStackDepth++].choosingRole);
            break;
          }
        }
      }
      else if (!isTerminal() && !mLatches.scoresAreLatched(lastInternalSetState))
      {
        if (rolloutStackDepth++ >= hintMoveDepth)
        {
          hintMove = null;
        }
      }
      else if (rolloutDecisionStack[rolloutStackDepth].nextChoiceIndex != rolloutDecisionStack[rolloutStackDepth].baseChoiceIndex)
      {
        //	Having recorded the potential terminal state continue to explore another
        //	branch given that this terminality was not a forced win for the deciding player
        RolloutDecisionState decisionState = rolloutDecisionStack[rolloutStackDepth];

        setPropNetUsage(decisionState.state);
        setBasePropositionsFromState(decisionState.state);
      }
      else if ( rolloutStackDepth > 0 &&
                (rolloutDecisionStack[rolloutStackDepth].chooserIndex == -1 ||
                 rolloutDecisionStack[rolloutStackDepth].chooserIndex == rolloutDecisionStack[rolloutStackDepth-1].chooserIndex) )
      {
        //  No choice lead to a win.  If there was a choice at the previous level and the
        //  result is bad for that player should pop and retry
        if ( rolloutDecisionStack[rolloutStackDepth - 1].nextChoiceIndex != rolloutDecisionStack[rolloutStackDepth - 1].baseChoiceIndex)
        {
          if (rolloutDecisionStack[rolloutStackDepth - 1].chooserIndex != -1)
          {
            int lScoreForChoosingRole = getGoal(roles[rolloutDecisionStack[rolloutStackDepth - 1].chooserIndex]);

            getLatchedScoreRange(lastInternalSetState,
                                 roles[rolloutDecisionStack[rolloutStackDepth - 1].chooserIndex],
                                 parentLatchedScoreRangeBuffer);
            if ( lScoreForChoosingRole == parentLatchedScoreRangeBuffer[0] )
            {
              rolloutDecisionStack[rolloutStackDepth].chooserMoves = null;

              RolloutDecisionState poppedState = rolloutDecisionStack[--rolloutStackDepth];

              setPropNetUsage(poppedState.state);
              setBasePropositionsFromState(poppedState.state);
              continue;
            }
          }
        }

        rolloutStackDepth++;
        break;
      }
      else
      {
        rolloutStackDepth++;
        break;
      }
    }
    while (cutoffDepth > rolloutStackDepth);
  }

  private double recursiveGreedyRollout(TerminalResultSet results,
                                        Factor factor,
                                        MoveWeights moveWeights,
                                        ForwardDeadReckonLegalMoveInfo[] playedMoves,
                                        int cutoffDepth)
  {
    rolloutSeq++;
    rolloutStackDepth = 0;
    totalRoleoutChoices = 0;
    totalRoleoutNodesExamined = 0;

    doRecursiveGreedyRoleout(results, factor, moveWeights, playedMoves, cutoffDepth);

    if (totalRoleoutNodesExamined > 0)
    {
      return totalRoleoutChoices / totalRoleoutNodesExamined;
    }
    return 0;
  }

  private Set<ForwardDeadReckonProposition> terminatingMoveProps             = new HashSet<>();
  public long                               numRolloutDecisionNodeExpansions = 0;
  public double                             greedyRolloutEffectiveness       = 0;
  private int                               terminalCheckHorizon             = 500; //  Effectively infinite by default

  public int getNumTerminatingMoveProps()
  {
    return terminatingMoveProps.size();
  }

  public void clearTerminatingMoveProps()
  {
    terminatingMoveProps.clear();
  }

  public void setTerminalCheckHorizon(int horizon)
  {
    terminalCheckHorizon = horizon;
  }

  private Move transitionToNextStateInGreedyRollout(TerminalResultSet results,
                                                    Factor factor,
                                                    Move hintMove,
                                                    MoveWeights moveWeights,
                                                    ForwardDeadReckonLegalMoveInfo[] playedMoves,
                                                    int moveIndex)
  {
    //		ProfileSection methodSection = new ProfileSection("TestPropnetStateMachine.transitionToNextStateInGreedyRollout");
    //		try
    //		{
    ForwardDeadReckonLegalMoveSet activeLegalMoves = propNet.getActiveLegalProps(instanceId);
    int index = 0;
    boolean simultaneousMove = false;
    int maxChoices = 0;
    ForwardDeadReckonLegalMoveInfo choicelessMoveInfo = null;

    RolloutDecisionState decisionState = rolloutDecisionStack[rolloutStackDepth];
    if (decisionState.rolloutSeq != rolloutSeq)
    {
      decisionState.rolloutSeq = rolloutSeq;
      decisionState.clearMoveChoices();
    }

    if (decisionState.chooserMoves == null)
    {
      decisionState.choosingRole = null;
      decisionState.chooserIndex = -1;
      decisionState.baseChoiceIndex = -1;
      decisionState.nextChoiceIndex = -1;
      decisionState.maxAchievableOpponentScoreTotal = -1;
      totalRoleoutNodesExamined++;

      for (Role role : getRoles())
      {
        ForwardDeadReckonLegalMoveSet moves = activeLegalMoves;
        int numChoices = StateMachineFilterUtils.getFilteredSize(null, moves, role, factor, false);

        if (numChoices > maxChoices)
        {
          maxChoices = numChoices;
        }

        if (numChoices > 1)
        {
          totalRoleoutChoices += numChoices;
          if (decisionState.choosingRole == null)
          {
            if (!simultaneousMove)
            {
              decisionState.choosingRole = role;
              decisionState.setNumMoveChoices(numChoices);
            }
          }
          else
          {
            int rand = getRandom(decisionState.numChoices);

            ForwardDeadReckonLegalMoveInfo info = decisionState.chooserMoves[rand];
            decisionState.nonChooserProps[decisionState.chooserIndex] = info.mInputProposition;
            if (playedMoves != null)
            {
              playedMoves[moveIndex] = info;
            }

            decisionState.choosingRole = null;
            decisionState.clearMoveChoices();
            simultaneousMove = true;
          }
        }

        if (simultaneousMove)
        {
          int rand = getRandom(numChoices);

          Iterator<ForwardDeadReckonLegalMoveInfo> itr = moves.getContents(role).iterator();
          for (int iMove = 0; iMove < numChoices; iMove++)
          {
            // Get next move for this factor
            ForwardDeadReckonLegalMoveInfo info = StateMachineFilterUtils.nextFilteredMove(factor, itr);

            if (rand-- <= 0)
            {
              decisionState.nonChooserProps[index++] = info.mInputProposition;
              //chosenJointMoveProps[index++] = info.inputProposition;
              if (playedMoves != null)
              {
                playedMoves[moveIndex] = info;
              }
              break;
            }
          }
        }
        else
        {
          int chooserMoveIndex = 0;
          Iterator<ForwardDeadReckonLegalMoveInfo> itr = moves.getContents(role).iterator();
          for (int iMove = 0; iMove < numChoices; iMove++)
          {
            // Get next move for this factor
            ForwardDeadReckonLegalMoveInfo info = StateMachineFilterUtils.nextFilteredMove(factor, itr);

            if (decisionState.choosingRole == role)
            {
              if (chooserMoveIndex == 0)
              {
                decisionState.chooserIndex = index++;
              }
              decisionState.chooserMoves[chooserMoveIndex++] = info;
            }
            else
            {
              decisionState.nonChooserProps[index++] = info.mInputProposition;
              if (info.mInputProposition != null || choicelessMoveInfo == null)
              {
                choicelessMoveInfo = info;
              }
              break;
            }
          }
        }
      }
    }

    if (simultaneousMove)
    {
      for(int i = 0; i < chosenJointMoveProps.length; i++)
      {
        chosenJointMoveProps[i] = decisionState.nonChooserProps[i];
      }

      transitionToNextStateFromChosenMove();

      if (isTerminal() || scoresAreLatched(lastInternalSetState))
      {
        results.considerResult(null);
      }
    }
    else if (decisionState.chooserIndex != -1)
    {
      int choiceIndex;
      boolean preEnumerate = mLatches.hasNegativelyLatchedGoals();
      int numTerminals = 0;

      if (decisionState.baseChoiceIndex == -1)
      {
        double total = 0;

        decisionState.state.copy(lastInternalSetState);
        decisionState.maxAchievableOpponentScoreTotal = 0;

        for(Role role : getRoles())
        {
          if ( !role.equals(decisionState.choosingRole))
          {
            getLatchedScoreRange(decisionState.state, role, latchedScoreRangeBuffer);

            decisionState.maxAchievableOpponentScoreTotal += latchedScoreRangeBuffer[1];
          }
        }

        for (int i = 0; i < decisionState.numChoices; i++)
        {
          ForwardDeadReckonLegalMoveInfo chooserMove = decisionState.chooserMoves[i];
          if (moveWeights != null)
          {
            total += moveWeights.weightScore[chooserMove.mMasterIndex];
          }
          if (!preEnumerate && terminatingMoveProps.contains(chooserMove.mInputProposition))
          {
            preEnumerate = true;
            numRolloutDecisionNodeExpansions++;
            if (moveWeights == null)
            {
              break;
            }
          }
        }

        if (moveWeights == null)
        {
          decisionState.baseChoiceIndex = getRandom(decisionState.numChoices);
        }
        else
        {
          total = getRandom((int)total);
        }

        for (int i = 0; i < decisionState.numChoices; i++)
        {
          decisionState.propProcessed[i] = false;
          if (decisionState.baseChoiceIndex == -1)
          {
            total -= moveWeights.weightScore[decisionState.chooserMoves[i].mMasterIndex];
            if (total <= 0)
            {
              decisionState.baseChoiceIndex = i;
            }
          }
        }

        choiceIndex = decisionState.baseChoiceIndex;
      }
      else
      {
        choiceIndex = decisionState.nextChoiceIndex;
      }

      for (int roleIndex = 0; roleIndex < getRoles().length; roleIndex++)
      {
        if (roleIndex != decisionState.chooserIndex)
        {
          chosenJointMoveProps[roleIndex] = decisionState.nonChooserProps[roleIndex];
        }
      }

      boolean transitioned = false;

      getLatchedScoreRange(lastInternalSetState, decisionState.choosingRole, parentLatchedScoreRangeBuffer);

      //	If we're given a hint move to check for a win do that first
      //	the first time we look at this node
      if (hintMove != null && decisionState.numChoices > 1)
      {
        if (decisionState.baseChoiceIndex == choiceIndex)
        {
          for (int i = 0; i < decisionState.numChoices; i++)
          {
            if (decisionState.chooserMoves[i].mMove == hintMove)
            {
              chosenJointMoveProps[decisionState.chooserIndex] = decisionState.chooserMoves[i].mInputProposition;

              transitionToNextStateFromChosenMove();

              if (isTerminal() || mLatches.scoresAreLatched(lastInternalSetState))
              {
                numTerminals++;

                if (getGoal(decisionState.choosingRole) == parentLatchedScoreRangeBuffer[1])
                {
                  if (playedMoves != null)
                  {
                    assert(decisionState.chooserMoves[i] != null);
                    playedMoves[moveIndex] = decisionState.chooserMoves[i];
                  }
                  greedyRolloutEffectiveness++;
                  //	If we have a choosable win stop searching
                  return hintMove;
                }

                results.considerResult(decisionState.choosingRole);

                decisionState.propProcessed[i] = true;
              }
              else if (mLatches.hasNegativelyLatchedGoals())
              {
                int newMaxAchievableOpponentScoreTotal = 0;
                for(Role role : getRoles())
                {
                  if ( !role.equals(decisionState.choosingRole))
                  {
                    getLatchedScoreRange(lastInternalSetState, role, latchedScoreRangeBuffer);

                    newMaxAchievableOpponentScoreTotal += latchedScoreRangeBuffer[1];
                  }
                }

                if ( newMaxAchievableOpponentScoreTotal < decisionState.maxAchievableOpponentScoreTotal )
                {
                  if ( getRandom(100) < latchImprovementWeight )
                  {
                    if (playedMoves != null)
                    {
                      assert(decisionState.chooserMoves[i] != null);
                      playedMoves[moveIndex] = decisionState.chooserMoves[i];
                    }

                    decisionState.nextChoiceIndex = decisionState.baseChoiceIndex;
                    if ( getRandom(100) < latchWorseningAvoidanceWeight )
                    {
                      return hintMove;
                    }
                    return null;
                  }
                }
              }

              transitioned = true;
              break;
            }
          }
        }
        else
        {
          //	Not the first time we've looked at this node
          hintMove = null;
        }
      }

      //	First time we visit the node try them all.  After that if we're asked to reconsider
      //	just take the next one from the last one we chose
      int remainingMoves = (decisionState.baseChoiceIndex == choiceIndex &&
                            preEnumerate ? decisionState.numChoices
                                        : 1);
      int choice = -1;
      int lastTransitionChoice = -1;

      for (int i = remainingMoves-1; i >= 0; i--)
      {
        choice = (i + choiceIndex) % decisionState.numChoices;

        //	Don't re-process the hint move that we looked at first unless this is the specific requested
        //  move
        if ( i > 0 )
        {
          if (decisionState.propProcessed[choice] ||
              hintMove == decisionState.chooserMoves[choice].mMove ||
              (preEnumerate && !terminatingMoveProps.contains(decisionState.chooserMoves[choice].mInputProposition)))
          {
            continue;
          }
        }

        if (transitioned)
        {
          setPropNetUsage(decisionState.state);
          setBasePropositionsFromState(decisionState.state);
        }

        chosenJointMoveProps[decisionState.chooserIndex] = decisionState.chooserMoves[choice].mInputProposition;
        lastTransitionChoice = choice;

        transitionToNextStateFromChosenMove();

        transitioned = true;

        if (isTerminal() || mLatches.scoresAreLatched(lastInternalSetState))
        {
          numTerminals++;

          if ( rolloutStackDepth <= terminalCheckHorizon )
          {
            terminatingMoveProps
                .add(decisionState.chooserMoves[choice].mInputProposition);
          }

          if (getGoal(decisionState.choosingRole) == parentLatchedScoreRangeBuffer[1])
          {
            if (playedMoves != null)
            {
              assert(decisionState.chooserMoves[choice] != null);
              playedMoves[moveIndex] = decisionState.chooserMoves[choice];
            }
            if (preEnumerate)
            {
              greedyRolloutEffectiveness++;
            }

            //	If we have a choosable win stop searching
            return decisionState.chooserMoves[choice].mMove;
          }

          results.considerResult(decisionState.choosingRole);
          decisionState.propProcessed[choice] = true;
        }
        else if (mLatches.hasNegativelyLatchedGoals())
        {
          int newMaxAchievableOpponentScoreTotal = 0;
          for(Role role : getRoles())
          {
            if ( !role.equals(decisionState.choosingRole))
            {
              getLatchedScoreRange(lastInternalSetState, role, latchedScoreRangeBuffer);

              newMaxAchievableOpponentScoreTotal += latchedScoreRangeBuffer[1];
            }
          }

          if ( newMaxAchievableOpponentScoreTotal < decisionState.maxAchievableOpponentScoreTotal )
          {
            if ( getRandom(100) < latchImprovementWeight )
            {
              if (playedMoves != null)
              {
                assert(decisionState.chooserMoves[choice] != null);
                playedMoves[moveIndex] = decisionState.chooserMoves[choice];
              }
              decisionState.nextChoiceIndex = (choiceIndex+1)%decisionState.numChoices;
              if ( getRandom(100) < latchWorseningAvoidanceWeight )
              {
                return decisionState.chooserMoves[choice].mMove;
              }
              return null;
            }
          }
        }
      }

      if ( !transitioned )
      {
        chosenJointMoveProps[decisionState.chooserIndex] = decisionState.chooserMoves[choice].mInputProposition;
        lastTransitionChoice = choice;

        transitionToNextStateFromChosenMove();
      }

      if (playedMoves != null)
      {
        assert(decisionState.chooserMoves[lastTransitionChoice] != null);
        playedMoves[moveIndex] = decisionState.chooserMoves[lastTransitionChoice];
      }

      decisionState.nextChoiceIndex = lastTransitionChoice;
      do
      {
        decisionState.nextChoiceIndex = (decisionState.nextChoiceIndex + 1) %
                                        decisionState.numChoices;
        if (!decisionState.propProcessed[decisionState.nextChoiceIndex] ||
            decisionState.nextChoiceIndex == decisionState.baseChoiceIndex)
        {
          break;
        }
      }
      while (decisionState.nextChoiceIndex != choiceIndex);

      if (preEnumerate && numTerminals > 0)
      {
        greedyRolloutEffectiveness += (decisionState.numChoices - numTerminals) /
                                      decisionState.numChoices;
      }
    }
    else
    {
      for (int roleIndex = 0; roleIndex < numRoles; roleIndex++)
      {
        chosenJointMoveProps[roleIndex] = decisionState.nonChooserProps[roleIndex];
      }
      transitionToNextStateFromChosenMove();

      if (playedMoves != null)
      {
        assert(choicelessMoveInfo != null);
        playedMoves[moveIndex] = choicelessMoveInfo;
      }
      if (isTerminal() || mLatches.scoresAreLatched(lastInternalSetState))
      {
        results.considerResult(decisionState.choosingRole);
      }
    }

    return null;
    //        }
    //		finally
    //		{
    //			methodSection.exitScope();
    //		}
  }

  private boolean hasAvailableMoveForAllRoles(ForwardDeadReckonPropNet net)
  {
    for (int roleIndex = 0; roleIndex < numRoles; roleIndex++)
    {
      Collection<ForwardDeadReckonLegalMoveInfo> moves = net.getActiveLegalProps(instanceId).getContents(roleIndex);

      if (moves.isEmpty())
      {
        return false;
      }
    }

    return true;
  }

  /**
   * Select a random joint move, according to the policy (if any), and execute it.  Record the played moves in the
   * provided array.
   *
   * @param factor         - the filter for the factor being played in (or null for the only factor).
   * @param playedMoves    - array to fill in with the chosen moves.
   * @param statesVisited
   *
   * @return the maximum number of choices of any role.
   */
  private int transitionToRandomJointMove(StateMachineFilter factor,
                                          ForwardDeadReckonLegalMoveInfo[] playedMoves,
                                          ForwardDeadReckonInternalMachineState[] statesVisited)
  {
		int result = 0;
    int numChoices;
    int choiceIndex = -1;
    int choosingRole = -1;
    boolean choiceSeen = false;
    boolean acceptableChoice = true;
    ForwardDeadReckonLegalMoveSet activeLegalMoves = getLegalMoveSet(lastInternalSetState);
    int startingDepth = rolloutDepth;
    boolean subtreeFullySearched = false;
    boolean lSimultaneousChoices = false;

    if ( mPlayoutPolicy != null )
    {
      mPlayoutPolicy.noteCurrentState(statesVisited == null ? lastInternalSetState : statesVisited[rolloutDepth], activeLegalMoves, factor, rolloutDepth, playedMoves, statesVisited);
    }

    // When playing with a non-default policy, we might need to chose moves multiple times (if there was stack-popping
    // involved).
    do
    {
      int index = 0;
      int numChooserChoices = 1;
      ForwardDeadReckonLegalMoveInfo chooserChoice = null;
      boolean policySelectedMove = false;

      // Choose a move for every role.
      for (int roleIndex = 0; roleIndex < numRoles; roleIndex++)
      {
        ForwardDeadReckonLegalMoveSet moves = activeLegalMoves;
        ForwardDeadReckonLegalMoveInfo chosen = null;
        Iterator<ForwardDeadReckonLegalMoveInfo> itr;

        if ( mPlayoutPolicy != null && playedMoves != null )
        {
          chosen = mPlayoutPolicy.selectMove(roleIndex);
        }

        if (chosen == null)
        {
          if (factor == null)
          {
            // In the default case, we'll just select between all legal moves.
            numChoices = moves.getNumChoices(roleIndex);
            itr = moves.getContents(roleIndex).iterator();
          }
          else
          {
            // In a factored game the terminal logic can sometimes span the factors in a way we don't cleanly cater for
            // currently, so use lack of legal moves as a proxy for terminality.
            if (moves.getNumChoices(roleIndex) == 0)
            {
              return 0;
            }

            numChoices = StateMachineFilterUtils.getFilteredSize(null, moves, roleIndex, factor, false);

            itr = moves.getContents(roleIndex).iterator();
          }

          // Keep a high-water mark of the number of choices for any role.  We'll return this to the caller.
          if (numChoices > result)
          {
            result = numChoices;
          }

          int moveIndex;
          if (numChoices > 1)
          {
            if ((choosingRole != -1 && choosingRole != roleIndex) || (lSimultaneousChoices))
            {
              // Multiple roles have choices so this must be a simultaneous move game which we do not currently support
              // playout policies in, and for which we must independently select moves for each role.
              choosingRole = -1;
              choiceIndex = -1;
              lSimultaneousChoices = true;
            }
            else
            {
              // We've found a role with a choice.  Assume, for now, that it's the only role with a choice.  (If another
              // role turns out to have a choice, we'll go through the branch above and clear the choosingRole index.)
              choosingRole = roleIndex;
              numChooserChoices = numChoices;
            }

            if (choiceIndex == -1)
            {
              choiceIndex = getRandom(numChoices);

              if (playoutStackMoveInitialChoiceIndex != null)
              {
                playoutStackMoveInitialChoiceIndex[rolloutDepth] = choiceIndex;
              }
            }
            assert(playoutStackMoveInitialChoiceIndex == null || playoutStackMoveInitialChoiceIndex[rolloutDepth] < numChoices);
            assert(choiceIndex < numChoices);

            moveIndex = choiceIndex;
          }
          else
          {
            // Select the only available move.
            moveIndex = 0;
            numChooserChoices = 1;

            if (playoutStackMoveInitialChoiceIndex != null)
            {
              playoutStackMoveInitialChoiceIndex[rolloutDepth] = 0;
            }
          }

          for (int iMove = 0; iMove < numChoices; iMove++)
          {
            ForwardDeadReckonLegalMoveInfo info;

            // Get next move for this factor
            info = StateMachineFilterUtils.nextFilteredMove(factor, itr);

            if (moveIndex == iMove)
            {
              chosen = info;
              if ( roleIndex == choosingRole )
              {
                chooserChoice = chosen;
              }
              break;
            }
          }
        }
        else
        {
          assert(moves.getContents(roleIndex).contains(chosen));
          policySelectedMove = true;
          numChoices = 2; //  Arbitrary number > 1
        }

        assert(chosen != null);
        if (validationMachine != null)
        {
          chosenMoves[index] = chosen.mMove;
        }
        ForwardDeadReckonProposition chosenMoveProp = chosen.mIsPseudoNoOp ? null : chosen.mInputProposition;
        chosenJointMoveProps[index++] = chosenMoveProp;
        if (playedMoves != null &&
            (numChoices > 1 ||
             (!choiceSeen &&
              ((chosenMoveProp != null && !chosen.mIsPseudoNoOp && !chosen.mIsVirtualNoOp) ||
               roleIndex == numRoles-1))))
        {
          playedMoves[rolloutDepth] = chosen;
          choiceSeen = true;
        }
      }

      transitionToNextStateFromChosenMove();

      //  Do we have a non-default policy that has not already selected an explicit move
      acceptableChoice = true;
      if ( !policySelectedMove && mPlayoutPolicy != null && playoutStackMoveInitialChoiceIndex != null && statesVisited != null && choosingRole != -1 )
      {
        assert(numChooserChoices > playoutStackMoveInitialChoiceIndex[rolloutDepth]);
        choiceIndex = (choiceIndex+1)%numChooserChoices;
        playoutStackMoveNextChoiceIndex[rolloutDepth] = choiceIndex;

        acceptableChoice = subtreeFullySearched ||
                           (mPlayoutPolicy.isAcceptableMove(chooserChoice, choosingRole) &&
                            mPlayoutPolicy.isAcceptableState(lastInternalSetState, choosingRole));

        if ( !acceptableChoice )
        {
          //LOGGER.info("Reject move " + chooserChoice + " at depth " + rolloutDepth);

          if ( choiceIndex == playoutStackMoveInitialChoiceIndex[rolloutDepth] )
          {
            while( choiceIndex == playoutStackMoveInitialChoiceIndex[rolloutDepth] )
            {
              if ( rolloutDepth > 0 &&
                   !subtreeFullySearched &&
                   mPlayoutPolicy.popStackOnAllUnacceptableMoves(startingDepth-rolloutDepth) )
              {
                //  Pop back up the stack
                rolloutDepth--;

                //LOGGER.info("Pop back to depth " + rolloutDepth);
                //System.out.println("Popped state is: " + statesVisited[rolloutDepth]);
                setPropNetUsage(statesVisited[rolloutDepth]);
                setBasePropositionsFromState(statesVisited[rolloutDepth]);
                activeLegalMoves = getLegalMoveSet(lastInternalSetState);
                choiceIndex = playoutStackMoveNextChoiceIndex[rolloutDepth];
                assert(choiceIndex < activeLegalMoves.getNumChoices(0));
                mPlayoutPolicy.noteCurrentState(statesVisited[rolloutDepth], activeLegalMoves, factor, rolloutDepth, playedMoves, statesVisited);
              }
              else
              {
                //LOGGER.info("Cannot pop any further");
                //  All possibilities have been explored so just play next path down to
                //  next required move depth (startingDepth chosen)
                subtreeFullySearched = true;
                //rolloutDepth = startingDepth;
                setPropNetUsage(statesVisited[rolloutDepth]);
                setBasePropositionsFromState(statesVisited[rolloutDepth]);
                activeLegalMoves = getLegalMoveSet(lastInternalSetState);
                choiceIndex = playoutStackMoveNextChoiceIndex[rolloutDepth];
                break;
              }
            }
          }
          else
          {
            setPropNetUsage(statesVisited[rolloutDepth]);
            setBasePropositionsFromState(statesVisited[rolloutDepth]);
            activeLegalMoves = getLegalMoveSet(lastInternalSetState);
          }
        }
        else
        {
          //LOGGER.info("Accept move " + chooserChoice + " at depth " + rolloutDepth);
        }
      }
      else if ( playoutStackMoveInitialChoiceIndex != null && playoutStackMoveNextChoiceIndex != null )
      {
        playoutStackMoveNextChoiceIndex[rolloutDepth] = playoutStackMoveInitialChoiceIndex[rolloutDepth];
      }
//      if ( numRoles == 1 && statesVisited != null && getNonControlMask() != null && choiceIndex != (startingRand+result-1)%result )
//      {
//        for(int i = 0; i <= rolloutDepth; i++)
//        {
//          maskStateBuffer.copy(lastInternalSetState);
//          maskStateBuffer.xor(statesVisited[i]);
//          maskStateBuffer.intersect(getNonControlMask());
//          if ( maskStateBuffer.size() == 0 )
//          {
//            acceptableChoice = false;
//            break;
//          }
//        }
//
//        if ( !acceptableChoice )
//        {
//          setPropNetUsage(statesVisited[rolloutDepth]);
//          setBasePropositionsFromState(statesVisited[rolloutDepth]);
//        }
//      }

      if ( rolloutDepth < startingDepth && acceptableChoice )
      {
        //  If we popped a move we need to continue down again until we fill in
        //  moves down to the expected starting level
        acceptableChoice = false;
        rolloutDepth++;
        choiceIndex = -1;
        assert(statesVisited != null);
        //System.out.println("Recording new state for depth " + rolloutDepth + ": " + lastInternalSetState);
        statesVisited[rolloutDepth].copy(lastInternalSetState);

        if(isTerminal())
        {
          break;
        }

        activeLegalMoves = getLegalMoveSet(lastInternalSetState);
        mPlayoutPolicy.noteCurrentState(statesVisited[rolloutDepth], activeLegalMoves, factor, rolloutDepth, playedMoves, statesVisited);
      }
    } while(!acceptableChoice);

    return result;
  }

  private class TerminalResultSet
  {
    int                                          mChoosingRoleIndex = -1;
    public int                                   mScoreForChoosingRole = -1;
    public ForwardDeadReckonInternalMachineState mState;

    public void considerResult(Role choosingRole)
    {
      if (mChoosingRoleIndex == -1)
      {
        // We haven't yet recorded the choosing role.  Do so now - if there is one.  There won't be one if either no
        // roles have a choice or more than one role has a choice in this state.
        if (choosingRole == null)
        {
          return;
        }

        for (mChoosingRoleIndex = 0; !roles[mChoosingRoleIndex].equals(choosingRole); mChoosingRoleIndex++) {/* Spin */}
      }

      // Would this result be chosen over the previous (if any)?
      int lScoreForChoosingRole = getGoal(roles[mChoosingRoleIndex]);
      if (mState == null || (lScoreForChoosingRole > mScoreForChoosingRole))
      {
        mScoreForChoosingRole = lScoreForChoosingRole;
        if (mState == null)
        {
          mState = new ForwardDeadReckonInternalMachineState(lastInternalSetState);
        }
        else
        {
          mState.copy(lastInternalSetState);
        }
      }
    }

    public void reset()
    {
      mChoosingRoleIndex = -1;
      mScoreForChoosingRole = -1;
    }
  }

  private int     rolloutDepth;
  private boolean enableGreedyRollouts = true;
  private boolean greedyRolloutsDisabledPersistently = false;
  private ForwardDeadReckonInternalMachineState maskStateBuffer = null;

  /**
   * @return whether greedy rollouts are enabled
   */
  public boolean getIsGreedyRollouts()
  {
    return enableGreedyRollouts;
  }

  /**
   * Whether to en/disable use of greedy rollouts (defaults to enabled for a newly constructed
   * state machine)
   * @param enabled       New state
   * @param persistently  Disablement may be persistent after which no further changes are allowed
   *                      This will perform irrevocable propnet optimizations for disabled greedy rollouts
   */
  public void enableGreedyRollouts(boolean enabled, boolean persistently)
  {
    //  Legal transitions check
    assert( !enabled || !greedyRolloutsDisabledPersistently );

    if ( enabled )
    {
      enableGreedyRollouts = true;
    }
    else
    {
      enableGreedyRollouts = false;

      if ( !persistently || greedyRolloutsDisabledPersistently )
      {
        return;
      }

      //  Switch to goalless networks
      propNetO = propNetOWithoutGoals;
      propNetX = propNetXWithoutGoals;

      if (instanceId == 0)
      {
        finalizePropositionCrossReferenceInfo();  //  Onto the goalless variants now

        propNetXWithoutGoals.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);
        propNetOWithoutGoals.crystalize(masterInfoSet, firstBasePropIndex, masterLegalMoveSet, maxInstances);

        for(ForwardDeadReckonPropositionInfo info : masterInfoSet)
        {
          ForwardDeadReckonPropositionCrossReferenceInfo crInfo = (ForwardDeadReckonPropositionCrossReferenceInfo)info;

          crInfo.xNetPropId = crInfo.xNetProp.id;
          crInfo.oNetPropId = crInfo.oNetProp.id;
        }

        propNetXWithoutGoals.reset(true);
        propNetOWithoutGoals.reset(true);

        //  Set move factor info and virtual-noop info
        setMoveInfoForPropnet(propNetXWithoutGoals);
        setMoveInfoForPropnet(propNetOWithoutGoals);
      }

      propNet = propNetX;
      propNetInstanceInfo = propNet.animator.getInstanceInfo(instanceId);
      lastInternalSetState = null;
      lastInternalSetStateX = null;
      lastInternalSetStateO = null;

      for(int i = 0; i < numRoles; i++)
      {
        previousMovePropsX[i] = null;
        previousMovePropsO[i] = null;
      }
    }
  }

  private int totalNumMoves = 0;

  public MoveWeights createMoveWeights()
  {
    return new MoveWeights(totalNumMoves, getRoles().length);
  }

  public ForwardDeadReckonInternalMachineState getCurrentState()
  {
    return lastInternalSetState;
  }

  public void getDepthChargeResult(ForwardDeadReckonInternalMachineState state,
                                   PlayoutInfo info)
  {
    rolloutDepth = 0;
    boolean lUseGreedyRollouts = enableGreedyRollouts && (numRoles <= 2);

    for (int i = 0; i < numRoles; i++)
    {
      ForwardDeadReckonProposition xProp = previousMovePropsX[i];
      ForwardDeadReckonProposition oProp = previousMovePropsO[i];

      if ( xProp != null )
      {
        previousMovePropsX[i] = null;
        propNetX.animator.getInstanceInfo(instanceId).changeComponentValueTo(xProp.id, false);
      }
      if ( oProp != null )
      {
        previousMovePropsO[i] = null;
        propNetO.animator.getInstanceInfo(instanceId).changeComponentValueTo(oProp.id, false);
      }
    }

    if (validationMachine != null)
    {
      validationState = state.getMachineState();
    }
    setPropNetUsage(state);
    setBasePropositionsFromState(state);
    for (int i = 0; i < numRoles; i++)
    {
      previouslyChosenJointMovePropIdsX[i] = -1;
      previouslyChosenJointMovePropIdsO[i] = -1;
    }
    if (!lUseGreedyRollouts)
    {
      int totalChoices = 0;

      if ( mPlayoutPolicy != null && info.statesVisited != null )
      {
        if ( mMaster.mTurnNumber != mLastPlayoutTurnNumber )
        {
          mPlayoutPolicy.noteNewTurn();
          mLastPlayoutTurnNumber = mMaster.mTurnNumber;
        }
        mPlayoutPolicy.noteNewPlayout();
        if ( mPlayoutPolicy.requiresMoveHistory() )
        {
          info.recordTrace = true;
        }
        if ( mPlayoutPolicy.requiresStateHistory() )
        {
          info.recordTraceStates = true;
          if ( playoutStackMoveInitialChoiceIndex == null )
          {
            playoutStackMoveInitialChoiceIndex = new int[info.playoutTrace.length];
            playoutStackMoveNextChoiceIndex = new int[info.playoutTrace.length];
          }
        }
      }

      if ( info.recordTraceStates )
      {
        info.statesVisited[0].copy(state);
      }

      while (!isTerminal() && !scoresAreLatched(lastInternalSetState) && (mPlayoutPolicy == null || !mPlayoutPolicy.terminatePlayout()))
      {
        int numChoices = transitionToRandomJointMove(info.factor, info.playoutTrace, info.statesVisited);
        totalChoices += numChoices;
        rolloutDepth++;
        if ( info.recordTraceStates )
        {
          info.statesVisited[rolloutDepth].copy(lastInternalSetState);
        }
        if ( rolloutDepth > info.cutoffDepth )
        {
          break;
        }
      }

      if ( mPlayoutPolicy != null )
      {
        mPlayoutPolicy.noteCompletePlayout(rolloutDepth, info.playoutTrace, info.statesVisited);
      }

      info.playoutLength = rolloutDepth;
      if ( rolloutDepth > 0 )
      {
        info.averageBranchingFactor = (totalChoices + rolloutDepth / 2) / rolloutDepth;
      }
      else
      {
        info.averageBranchingFactor = 0;
      }
    }
    else
    {
      double branchingFactor = 0;

      if ( !isTerminal() )
      {
        mResultSet.reset();
        branchingFactor = recursiveGreedyRollout(mResultSet,
                                                 info.factor,
                                                 info.moveWeights,
                                                 info.playoutTrace,
                                                 info.cutoffDepth);

        if (mResultSet.mChoosingRoleIndex != -1)
        {
          setPropNetUsage(mResultSet.mState);
          setBasePropositionsFromState(mResultSet.mState);
        }

        rolloutDepth = rolloutStackDepth;
      }

      assert(rolloutStackDepth>0);
      info.playoutLength = rolloutStackDepth;
      info.averageBranchingFactor = (int)(branchingFactor + 0.5);
    }
    for (int i = 0; i < numRoles; i++)
    {
      int xId = previouslyChosenJointMovePropIdsX[i];
      int oId = previouslyChosenJointMovePropIdsO[i];

      if ( xId != -1)
      {
        propNetX.animator.getInstanceInfo(instanceId).changeComponentValueTo(xId, false);
      }
      if ( oId != -1)
      {
        propNetO.animator.getInstanceInfo(instanceId).changeComponentValueTo(oId, false);
      }
    }
  }

  public Set<Factor> getFactors()
  {
    return factors;
  }

  public Set<GdlSentence> getBasePropositions()
  {
    return fullPropNet.getBasePropositions().keySet();
  }

  public PolymorphicProposition getBaseProposition(int xiIndex)
  {
    return fullPropNet.getBasePropositionsArray()[xiIndex];
  }

  private ForwardDeadReckonInternalMachineState lastGoalState = null;

  public void setGoalsCalculator(GoalsCalculator calculator)
  {
    mGoalsCalculator = calculator;
  }

  /**
   * @param playoutPolicy - policy to be used for playouts made with this state machine instance
   */
  public void setPlayoutPolicy(IPlayoutPolicy playoutPolicy)
  {
    LOGGER.info("Setting playout policy to " + playoutPolicy);
    mPlayoutPolicy = playoutPolicy;
  }

  public IPlayoutPolicy getPlayoutPolicy()
  {
    return mPlayoutPolicy;
  }

  private void setGoalNetBasePropsFromState(ForwardDeadReckonInternalMachineState state)
  {
    InternalMachineStateIterator lIterator = mStateIterator;
    ForwardDeadReckonPropnetFastAnimator.InstanceInfo instanceInfo = goalsNet.animator.getInstanceInfo(instanceId);

    if (lastGoalState != null)
    {
      if (!lastGoalState.equals(state))
      {
        lastGoalState.xor(state);

        if ( removeOldBasePropsBeforeAddingNew )
        {
          lIterator.reset(lastGoalState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.goalsNetProp != null && infoCr.goalsNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && !state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.goalsNetProp.id, false);
            }
          }
          lIterator.reset(lastGoalState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.goalsNetProp != null && infoCr.goalsNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.goalsNetProp.id, true);
            }
          }
        }
        else
        {
          lIterator.reset(lastGoalState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.goalsNetProp != null && infoCr.goalsNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.goalsNetProp.id, true);
            }
          }
          lIterator.reset(lastGoalState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.goalsNetProp != null && infoCr.goalsNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && !state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.goalsNetProp.id, false);
            }
          }
        }

        lastGoalState.copy(state);
      }
    }
    else
    {
      lIterator.reset(state);
      while (lIterator.hasNext())
      {
        ForwardDeadReckonPropositionInfo s = lIterator.next();
        ForwardDeadReckonPropositionCrossReferenceInfo sCr = (ForwardDeadReckonPropositionCrossReferenceInfo)s;
        if (sCr.goalsNetProp != null && sCr.goalsNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId)
        {
          if (state.contains(sCr))
          {
            instanceInfo.changeComponentValueTo(sCr.goalsNetProp.id, true);
          }
        }
      }

      lastGoalState = new ForwardDeadReckonInternalMachineState(state);
    }
  }

  private ForwardDeadReckonInternalMachineState lastTerminalityNetState = null;

  private void setTerminalityNetBasePropsFromState(ForwardDeadReckonInternalMachineState state)
  {
    InternalMachineStateIterator lIterator = mStateIterator;
    ForwardDeadReckonPropnetFastAnimator.InstanceInfo instanceInfo = terminalityNet.animator.getInstanceInfo(instanceId);

    if (lastTerminalityNetState != null)
    {
      if (!lastTerminalityNetState.equals(state))
      {
        lastTerminalityNetState.xor(state);

        if ( removeOldBasePropsBeforeAddingNew )
        {
          lIterator.reset(lastTerminalityNetState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.terminalityNetProp != null && infoCr.terminalityNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && !state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.terminalityNetProp.id, false);
            }
          }
          lIterator.reset(lastTerminalityNetState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.terminalityNetProp != null && infoCr.terminalityNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.terminalityNetProp.id, true);
            }
          }
        }
        else
        {
          lIterator.reset(lastTerminalityNetState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.terminalityNetProp != null && infoCr.terminalityNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.terminalityNetProp.id, true);
            }
          }
          lIterator.reset(lastTerminalityNetState);
          while (lIterator.hasNext())
          {
            ForwardDeadReckonPropositionInfo info = lIterator.next();
            ForwardDeadReckonPropositionCrossReferenceInfo infoCr = (ForwardDeadReckonPropositionCrossReferenceInfo)info;
            if ( infoCr.terminalityNetProp != null && infoCr.terminalityNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId && !state.contains(info) )
            {
              instanceInfo.changeComponentValueTo(infoCr.terminalityNetProp.id, false);
            }
          }
        }

        lastTerminalityNetState.copy(state);
      }
    }
    else
    {
      lIterator.reset(state);
      while (lIterator.hasNext())
      {
        ForwardDeadReckonPropositionInfo s = lIterator.next();
        ForwardDeadReckonPropositionCrossReferenceInfo sCr = (ForwardDeadReckonPropositionCrossReferenceInfo)s;
        if (sCr.terminalityNetProp != null && sCr.terminalityNetProp.id != ForwardDeadReckonPropnetFastAnimator.notNeededComponentId)
        {
          if (state.contains(sCr))
          {
            instanceInfo.changeComponentValueTo(sCr.terminalityNetProp.id, true);
          }
        }
      }

      lastTerminalityNetState = new ForwardDeadReckonInternalMachineState(state);
    }
  }

  public int getGoal(ForwardDeadReckonInternalMachineState state, Role role)
  {
    if ( mGoalsCalculator != null )
    {
      return mGoalsCalculator.getGoalValue(state == null ? lastInternalSetState : state, role);
    }

    ForwardDeadReckonPropNet net;

    if (enableGreedyRollouts)
    {
      if (state != null)
      {
        setPropNetUsage(state);
        setBasePropositionsFromState(state);
      }

      net = propNet;
    }
    else
    {
      net = goalsNet;

      if (state == null)
      {
        state = lastInternalSetState;
      }

      setGoalNetBasePropsFromState(state);

//      if (lastGoalState == null)
//      {
//        for (PolymorphicProposition p : net.getBasePropositionsArray())
//        {
//          net.setProposition(instanceId, (ForwardDeadReckonProposition)p, false);
//          //((ForwardDeadReckonProposition)p).setValue(false, instanceId);
//        }
//
//        lIterator.reset(state);
//        while (lIterator.hasNext())
//        {
//          ForwardDeadReckonPropositionInfo s = lIterator.next();
//          ForwardDeadReckonPropositionCrossReferenceInfo scr = (ForwardDeadReckonPropositionCrossReferenceInfo)s;
//          if (scr.goalsNetProp != null)
//          {
//            net.setProposition(instanceId, scr.goalsNetProp, true);
//            //scr.goalsNetProp.setValue(true, instanceId);
//          }
//        }
//
//        if (lastGoalState == null)
//        {
//          lastGoalState = new ForwardDeadReckonInternalMachineState(state);
//        }
//        else
//        {
//          lastGoalState.copy(state);
//        }
//      }
//      else if (!state.equals(lastGoalState))
//      {
//        if (nextGoalState == null)
//        {
//          nextGoalState = new ForwardDeadReckonInternalMachineState(state);
//        }
//        else
//        {
//          nextGoalState.copy(state);
//        }
//
//        lastGoalState.xor(state);
//
//        lIterator.reset(lastGoalState);
//        while (lIterator.hasNext())
//        {
//          ForwardDeadReckonPropositionInfo info = lIterator.next();
//          ForwardDeadReckonProposition goalsNetProp = ((ForwardDeadReckonPropositionCrossReferenceInfo)info).goalsNetProp;
//          if (goalsNetProp != null)
//          {
//            if (nextGoalState.contains(info))
//            {
//              net.setProposition(instanceId, goalsNetProp, true);
//              //goalsNetProp.setValue(true, instanceId);
//            }
//            else
//            {
//              net.setProposition(instanceId, goalsNetProp, false);
//              //goalsNetProp.setValue(false, instanceId);
//            }
//          }
//        }
//
//        lastGoalState.copy(nextGoalState);
//      }


    }

    //  HACK - for factored games we might be determining terminality
    //  based on factor termination through lack of legal moves, but not
    //  actually have a complete game state that registers as terminal.
    //  In such cases the goal network may also rely on global terminality
    //  and we attempt to best-guess the actual goal in this case.  Specifically
    //  if the game is terminal in a factor but not globally terminal, and all
    //  roles report the same score, we ASSUME it should be a normalized draw
    //  with all scoring 50.
    if ( factors != null && !isTerminalUnfactored() && isTerminal() )
    {
      int observedResult = -1;
      boolean goalDifferentiated = false;
      boolean roleSeen = false;

      for(Role r : getRoles())
      {
        int value = extractRoleGoal(net, role);

        if ( observedResult == -1 )
        {
          observedResult = value;
        }
        else if ( observedResult != value )
        {
          if ( roleSeen )
          {
            break;
          }

          observedResult = value;
          goalDifferentiated = true;
        }

        if ( role.equals(r) )
        {
          roleSeen = true;
          if ( goalDifferentiated )
          {
            break;
          }
        }
      }

      if ( !goalDifferentiated )
      {
        observedResult = 50;
      }

      return observedResult;
    }

    return extractRoleGoal(net, role);
  }

  private int extractRoleGoal(ForwardDeadReckonPropNet net, Role role)
  {
    PolymorphicProposition[] goalProps = net.getGoalPropositions().get(role);
    int result = 0;

    for (PolymorphicProposition p : goalProps)
    {
      if ( net.getActiveBaseProps(instanceId).contains(((ForwardDeadReckonProposition)p).getInfo()) )
      {
        result = Integer.parseInt(p.getName().getBody().get(1).toString());
        break;
      }
    }

    return result;
  }
}
