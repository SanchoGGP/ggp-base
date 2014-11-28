
package org.ggp.base.player.gamer.statemachine.sancho.heuristic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.player.gamer.statemachine.sancho.RoleOrdering;
import org.ggp.base.player.gamer.statemachine.sancho.TreeNode;
import org.ggp.base.util.gdl.grammar.GdlFunction;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropositionInfo;
import org.ggp.base.util.statemachine.MachineState;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine;
import org.ggp.base.util.stats.PearsonCorrelation;

/**
 * Heuristic which assumes that it's better to have more pieces than few.
 */
public class PieceHeuristic implements Heuristic
{
  /**
   * Logger instance
   */
  static final Logger LOGGER = LogManager.getLogger();

  private static final int                                               MIN_PIECE_PROP_ARITY      = 3;    // Assume board of at least 2 dimensions + piece type
  private static final int                                               MAX_PIECE_PROP_ARITY      = 4;    // For now (until we can do game-specific learning) restrict to exactly 2-d boards
  private static final int                                               MIN_PIECES_THRESHOLD      = 6;
  private static final double                                            MIN_HEURISTIC_CORRELATION = 0.06;
  //  Whether to use mobility values for pieces (else score-weighted aggregate presence)
  //  Empirically (in its current form at least) aggregate presence does not seem to work well
  private static final boolean                                           USE_MOBILITY_VALUES_FOR_PIECES = true;
  //  This is a bit of a hack (well, a lot of one really).  Empirically the metagaming tunes the piece values badly in
  //  the only known games with 2 piece types (checkers variants), so FOR NOW we disable differentiated piece
  //  type weights for <=2 piece types
  private static final int                                               MIN_NUM_PIECES            = 3;
  //  Similarly - we don't want to consider very large choices as possibly being piece types - for now we
  //  cap at what chess needs (6 types of pieces, possibly encoded separately for each of two roles)
  private static final int                                               MAX_NUM_PIECES            = 12;
  //  How much of a material difference do we consider enough to warrant sequence flagging
  private static final double                                            MIN_HEURISTIC_DIFF_FOR_SEQUENCE = 0.01;

  private static final double                                            EPSILON                   = 1e-6;

  private Map<PieceMaskSpecifier, HeuristicScoreInfo>                    propGroupScoreSets        = null;
  int                                                                    numRoles                  = 0;
  private PieceMaskSpecifier[]                                           pieceSets                 = null;
  private int                                                            totalSimulatedTurns       = 0;
  //  The following track runtime usage state and are dependent on the current game-state
  private TreeNode                                                       rootNode                  = null;
  private int                                                            heuristicSampleWeight     = 10;
  private double[]                                                       rootPieceValues           = null;
  private boolean                                                        mTuningComplete           = false;

  private static class GdlFunctionInfo
  {
    private String           name;
    public List<Set<String>> paramRanges;

    public GdlFunctionInfo(String xiName, int arity)
    {
      name = xiName;
      paramRanges = new ArrayList<>();

      for (int i = 0; i < arity; i++)
      {
        paramRanges.add(new HashSet<String>());
      }
    }

    public String getName()
    {
      return name;
    }
  }

  private static class PotentialPiecePropSet
  {
    String fnName;
    int    potentialRoleArgIndex;

    public PotentialPiecePropSet(String name, int index)
    {
      fnName = name;
      potentialRoleArgIndex = index;
    }
  }

  private static class HeuristicScoreInfo
  {
    private PearsonCorrelation[] roleCorrelation;
    int                          lastValue        = -1;
    boolean[]                    hasRoleChanges;
    double                       noChangeTurnRate = 0;

    public HeuristicScoreInfo(int numRoles)
    {
      roleCorrelation = new PearsonCorrelation[numRoles];
      hasRoleChanges = new boolean[numRoles];
      for (int i = 0; i < numRoles; i++)
      {
        roleCorrelation[i] = new PearsonCorrelation();
      }
    }

    public void accrueSample(int value, int[] roleValues)
    {
      for (int i = 0; i < roleCorrelation.length; i++)
      {
        roleCorrelation[i].sample(value, roleValues[i]);
      }
    }

    public double[] getRoleCorrelations()
    {
      double[] result = new double[roleCorrelation.length];

      for (int i = 0; i < roleCorrelation.length; i++)
      {
        result[i] = roleCorrelation[i].getCorrelation();
      }

      return result;
    }
  }

  private class ConstituentPieceInfo
  {
    final String name;
    final public ForwardDeadReckonInternalMachineState mask;
    public double value;

    public ConstituentPieceInfo(String xiName, ForwardDeadReckonInternalMachineState pieceMask)
    {
      name = xiName;
      mask = pieceMask;
      value = 1;
    }

    @Override
    public String toString()
    {
      return name + ":" + value;
    }
  }

  private class PieceUsageStats
  {
    private int aggregatePresenceInGame;
    private int maxPresenceInGame;
    private final double[] scoreWeightedPresence;
    private final double[] scoreWeightedUsage;
    private double usageInGame;
    private int usageCountInGame;
    private int numSampleGames;
    private int numSampleMovesInGame;
    private ForwardDeadReckonInternalMachineState previousMaskedState;
    private final ConstituentPieceInfo pieceInfo;

    public PieceUsageStats(ConstituentPieceInfo xiPieceInfo, int numRoles)
    {
      pieceInfo = xiPieceInfo;
      scoreWeightedPresence = new double[numRoles];
      scoreWeightedUsage = new double[numRoles];

      for(int i = 0; i < numRoles; i++)
      {
        scoreWeightedPresence[i] = 0;
        scoreWeightedUsage[i] = 0;
      }

      numSampleGames = 0;
    }

    public void startSampleGame()
    {
      aggregatePresenceInGame = 0;
      maxPresenceInGame = 0;
      numSampleMovesInGame = 0;
      usageCountInGame = 0;
      usageInGame = 0;

      previousMaskedState = null;
    }

    public void sampleMove(ForwardDeadReckonInternalMachineState xiState)
    {
      ForwardDeadReckonInternalMachineState maskedState = new ForwardDeadReckonInternalMachineState(xiState);
      maskedState.intersect(pieceInfo.mask);

      int presence = pieceInfo.mask.intersectionSize(xiState);

      if ( presence > maxPresenceInGame )
      {
        maxPresenceInGame = presence;
      }

      aggregatePresenceInGame += presence;
      numSampleMovesInGame++;

      if ( previousMaskedState != null &&
           maskedState.size() > 0 )
      {
        if ( previousMaskedState.size() == maskedState.size() &&
             previousMaskedState.intersectionSize(maskedState) != maskedState.size() )
        {
          //  A piece of this type has been moved
          usageInGame += (double)1/maskedState.size();
        }
        usageCountInGame++;
      }

      previousMaskedState = maskedState;
    }

    public void endSampleGame(int[] result)
    {
      double weightedPresence;
      double weightedUsage;
      double averagePresenceInGame = aggregatePresenceInGame/numSampleMovesInGame;

      if ( maxPresenceInGame > 0 )
      {
        weightedPresence = averagePresenceInGame/maxPresenceInGame;
      }
      else
      {
        weightedPresence = 0;
      }

      if ( usageCountInGame > 0 )
      {
        weightedUsage = usageInGame/usageCountInGame;
      }
      else
      {
        weightedUsage = 0;
      }

      for(int i = 0; i < result.length; i++)
      {
        double normalizedScore = ((double)result[i])/100;
        scoreWeightedPresence[i] = (scoreWeightedPresence[i]*numSampleGames + weightedPresence*normalizedScore)/(numSampleGames+1);
        scoreWeightedUsage[i] = (scoreWeightedUsage[i]*numSampleGames + weightedUsage*normalizedScore)/(numSampleGames+1);

        assert(!Double.isNaN(scoreWeightedUsage[i]));
        assert(!Double.isInfinite(scoreWeightedUsage[i]));
      }
      numSampleGames++;
    }

    public double getPresenceWeighting(int roleIndex)
    {
      return scoreWeightedPresence[roleIndex];
    }

    public double getMobilityWeighting(int roleIndex)
    {
      return scoreWeightedUsage[roleIndex];
    }

    public String getPieceName()
    {
      return pieceInfo.name;
    }

    public ConstituentPieceInfo getPieceInfo()
    {
      return pieceInfo;
    }
  }

  private class PieceMaskSpecifier
  {
    public final ForwardDeadReckonInternalMachineState  overallPieceMask;
    public ConstituentPieceInfo[]                       individualPieceTypes = null;
    public List<PieceUsageStats>                        individualPieceUsageStats = new ArrayList<>();

    public PieceMaskSpecifier(ForwardDeadReckonInternalMachineState pieceMask)
    {
      overallPieceMask = pieceMask;

      //  Can we see multiple different piece types?
      int propIndex = -1;

      Map<String, GdlFunctionInfo> basePropFns = new HashMap<>();

      while( (propIndex = overallPieceMask.getContents().nextSetBit(propIndex+1)) != -1 )
      {
        ForwardDeadReckonPropositionInfo prop = overallPieceMask.resolveIndex(propIndex);

        addSentenceToFnMap(prop.sentence, basePropFns);
      }

      Map<String, ForwardDeadReckonInternalMachineState> pieceTypeMasks = new HashMap<>();

      //  We only expect one piece fn name (e.g. 'cell' or similar which is the state of a board cell)
      //  If we have a more complex situation don't attempt to find piece types
      //  FUTURE - this may need further generalization
      if(basePropFns.size() == 1)
      {
        GdlFunctionInfo fnInfo = basePropFns.values().iterator().next();
        int smallestRangeSize = Integer.MAX_VALUE;
        int smallestRangeIndex = -1;
        int ignoreSize = -1;

        for (int i = 0; i < fnInfo.paramRanges.size(); i++)
        {
          int rangeSize = fnInfo.paramRanges.get(i).size();

          //  Special case - a range size of 1 could just be a role indicator
          //  so if the arity is sufficient that ignoring such a range still
          //  leaves sufficient arity for a piece prop ignore it
          if ( rangeSize > 1 || fnInfo.paramRanges.size() == MIN_PIECE_PROP_ARITY )
          {
            if ( rangeSize < smallestRangeSize )
            {
              smallestRangeSize = rangeSize;
            }
            else if ( rangeSize == smallestRangeSize )
            {
              ignoreSize = smallestRangeSize;
            }
          }
        }

        smallestRangeSize = Integer.MAX_VALUE;

        for(int rangeIndex = 0; rangeIndex < fnInfo.paramRanges.size(); rangeIndex++)
        {
          int rangeSize = fnInfo.paramRanges.get(rangeIndex).size();

          //  Special case - a range size of 1 could just be a role indicator
          //  so if the arity is sufficient that ignoring such a range still
          //  leaves sufficient arity for a piece prop ignore it
          //  MASSIVE HACK - if all the plausible ranges have the same size then this will be the ignore size
          //  and we cannot be sure (without much more analysis!) which is the plausible piece-type one.  For now we
          //  just take the last in such cases which tends to work because the GDL tends to specify coordinates then type
          //  However this is totally arbitrary and therefore a complete hack
          //  TODO - more advanced analysis needed of how props change to detect the difference between a corrdinate
          //  a a piece type qualifier more reliably
          if ( (rangeSize != ignoreSize || rangeIndex == fnInfo.paramRanges.size()-1) && rangeSize < smallestRangeSize && (rangeSize > 1 || fnInfo.paramRanges.size() == MIN_PIECE_PROP_ARITY))
          {
            smallestRangeSize = rangeSize;
            smallestRangeIndex = rangeIndex;
          }
        }

        //  Should validate that the different values of this arg actually
        //  plausibly correspond to pieces (by checking that moves which preserve
        //  the overall piece count predominantly change exactly 2 bits in the
        //  piece mask - only 'predominantly' because we don't want rare piece
        //  converting moves like pawns reaching the 8th rank in chess preventing
        //  recognition)
        //  TODO - write some code to do this validation

        //  Create masking info for each piece type
        while( (propIndex = overallPieceMask.getContents().nextSetBit(propIndex+1)) != -1 )
        {
          ForwardDeadReckonPropositionInfo prop = overallPieceMask.resolveIndex(propIndex);

          if (prop.sentence.arity() == 1 &&
              prop.sentence.getBody().get(0) instanceof GdlFunction)
          {
            GdlFunction propFn = (GdlFunction)prop.sentence.getBody().get(0);

            String pieceType = propFn.getBody().get(smallestRangeIndex).toString();
            ForwardDeadReckonInternalMachineState pieceTypeMask;

            if ( pieceTypeMasks.containsKey(pieceType))
            {
              pieceTypeMask = pieceTypeMasks.get(pieceType);
            }
            else
            {
              pieceTypeMask = new ForwardDeadReckonInternalMachineState(pieceMask);
              pieceTypeMask.clear();
              pieceTypeMasks.put(pieceType, pieceTypeMask);
            }

            pieceTypeMask.add(prop);
          }
        }

        assert(pieceTypeMasks.size() == smallestRangeSize);

        //if ( pieceTypeMasks.size() >= MIN_NUM_PIECES )
        {
          individualPieceTypes = new ConstituentPieceInfo[pieceTypeMasks.size()];

          int index = 0;
          for(Entry<String, ForwardDeadReckonInternalMachineState> entry : pieceTypeMasks.entrySet())
          {
            individualPieceTypes[index] = new ConstituentPieceInfo(entry.getKey(), entry.getValue());
            individualPieceUsageStats.add(new PieceUsageStats(individualPieceTypes[index], numRoles));
            index++;
          }
        }
      }
    }

    /**
     * Merge the info for two piece maps together to create an amalgamated one
     * @param other
     */
    public void merge(PieceMaskSpecifier other)
    {
      overallPieceMask.merge(other.overallPieceMask);

      int newPieceTypeCount = (individualPieceTypes == null ? 1 : individualPieceTypes.length) + (other.individualPieceTypes == null ? 1 : other.individualPieceTypes.length);

      if ( newPieceTypeCount > 0 )
      {
        ConstituentPieceInfo[] combined = new ConstituentPieceInfo[newPieceTypeCount];

        int index = 0;

        if ( individualPieceTypes != null )
        {
          for(ConstituentPieceInfo pieceInfo : individualPieceTypes)
          {
            combined[index++] = pieceInfo;
          }
        }

        if ( other.individualPieceTypes != null )
        {
          for(ConstituentPieceInfo pieceInfo : other.individualPieceTypes)
          {
            combined[index++] = pieceInfo;
          }
        }

        individualPieceTypes = combined;
        individualPieceUsageStats.addAll(other.individualPieceUsageStats);
      }
    }

    public void finalizePieceValues(int roleIndex)
    {
      if ( individualPieceTypes.length < MIN_NUM_PIECES )
      {
        individualPieceTypes = null;
      }

      if ( USE_MOBILITY_VALUES_FOR_PIECES )
      {
        double minExpMobilityMeasure = Double.MAX_VALUE;
        double mobilityNormalizer = 1;

        for(PieceUsageStats pieceUsage : individualPieceUsageStats)
        {
          double mobilityMeasure = pieceUsage.getMobilityWeighting(roleIndex);
          double presenceMeasure = pieceUsage.getPresenceWeighting(roleIndex);

          if ( presenceMeasure != 0 )
          {
            if ( mobilityMeasure < minExpMobilityMeasure )
            {
              //  If a piece has correlation but no mobility at all we cannot measure relative values
              //  so don't attempt to discriminate
              if ( mobilityMeasure == 0 )
              {
                individualPieceTypes = null;
              }

              minExpMobilityMeasure = mobilityMeasure;
              mobilityNormalizer = 1/minExpMobilityMeasure;
            }
          }
        }

        for(PieceUsageStats pieceUsage : individualPieceUsageStats)
        {
          LOGGER.info("Piece type " + pieceUsage.getPieceName() + " mobility value: " + pieceUsage.getMobilityWeighting(roleIndex)*mobilityNormalizer);

          pieceUsage.getPieceInfo().value = pieceUsage.getMobilityWeighting(roleIndex)*mobilityNormalizer;
        }
      }
      else
      {
        double minPresenceMeasure = Double.MAX_VALUE;
        double presenceNormalizer = 1;

        for(PieceUsageStats pieceUsage : individualPieceUsageStats)
        {
          double presenceMeasure = pieceUsage.getPresenceWeighting(roleIndex);

          if ( presenceMeasure < minPresenceMeasure )
          {
            minPresenceMeasure = presenceMeasure;
            presenceNormalizer = 1/minPresenceMeasure;
          }
        }

        for(PieceUsageStats pieceUsage : individualPieceUsageStats)
        {
          LOGGER.info("Piece type " + pieceUsage.getPieceName() + " presence value: " + pieceUsage.getPresenceWeighting(roleIndex)*presenceNormalizer);

          pieceUsage.getPieceInfo().value = pieceUsage.getPresenceWeighting(roleIndex)*presenceNormalizer;
        }
      }
    }

    /**
     * Return the material value of a state
     * @param state
     * @return material value according to the known piece types and values
     */
    public double getValue(ForwardDeadReckonInternalMachineState state)
    {
      double result = 0;

      if ( individualPieceTypes == null )
      {
        result += overallPieceMask.intersectionSize(state);
      }
      else
      {
        for(ConstituentPieceInfo pieceInfo : individualPieceTypes)
        {
          result += pieceInfo.mask.intersectionSize(state)*pieceInfo.value;
        }
      }
      return result;
    }

    @Override
    public String toString()
    {
      return overallPieceMask.toString() + " with piece types " + java.util.Arrays.toString(individualPieceTypes);
    }
  }

  /**
   * Default constructor
   */
  public PieceHeuristic()
  {
  }

  private PieceHeuristic(PieceHeuristic copyFrom)
  {
    propGroupScoreSets         = copyFrom.propGroupScoreSets;
    numRoles                   = copyFrom.numRoles;
    pieceSets                  = copyFrom.pieceSets;
    totalSimulatedTurns        = copyFrom.totalSimulatedTurns;
    //  The following track runtime usage state and are dependent on the current game-state
    rootPieceValues            = new double[numRoles];
  }

  /**
   * Add values of each argument of a sentences body to a fn value map
   * @param sentence
   * @param fnMap
   */
  static void addSentenceToFnMap(GdlSentence sentence, Map<String, GdlFunctionInfo> fnMap)
  {
    if (sentence.arity() == 1 &&
        sentence.getBody().get(0) instanceof GdlFunction)
    {
      GdlFunction propFn = (GdlFunction)sentence.getBody().get(0);

      if (propFn.arity() >= MIN_PIECE_PROP_ARITY &&
          propFn.arity() <= MAX_PIECE_PROP_ARITY)
      {
        GdlFunctionInfo fnInfo;

        if (fnMap.containsKey(propFn.getName().toString()))
        {
          fnInfo = fnMap.get(propFn.getName().toString());
        }
        else
        {
          fnInfo = new GdlFunctionInfo(propFn.getName().toString(),
                                       propFn.arity());
          fnMap.put(propFn.getName().toString(), fnInfo);
        }

        for (int i = 0; i < propFn.arity(); i++)
        {
          fnInfo.paramRanges.get(i).add(propFn.getBody().get(i).toString());
        }
      }
    }
  }

  @Override
  public boolean tuningInitialise(ForwardDeadReckonPropnetStateMachine stateMachine,
                               RoleOrdering xiRoleOrdering)
  {
    pieceSets = null;
    numRoles = stateMachine.getRoles().length;
    propGroupScoreSets = new HashMap<>();

    //  Currently we only support piece heuristics in 2 player games as we apply them
    //  in an assumed fixed-sum manner
    if ( numRoles != 2 )
    {
      return false;
    }

    rootPieceValues = new double[numRoles];

    Map<String, GdlFunctionInfo> basePropFns = new HashMap<>();

    for (GdlSentence baseProp : stateMachine.getBasePropositions())
    {
      addSentenceToFnMap(baseProp, basePropFns);
    }

    Set<PotentialPiecePropSet> potentialPiecePropSets = new HashSet<>();

    for (GdlFunctionInfo fnInfo : basePropFns.values())
    {
      int smallestFit = Integer.MAX_VALUE;
      int smallestFitIndex = -1;
      int ignoreSize = -1;

      for (int i = 0; i < fnInfo.paramRanges.size(); i++)
      {
        int numValues = fnInfo.paramRanges.get(i).size();

        if ( numValues % numRoles == 0 || numValues % numRoles == 1)
        {
          if ( numValues < smallestFit )
          {
            smallestFit = numValues;
          }
          else if ( numValues == smallestFit )
          {
            ignoreSize = smallestFit;
          }
        }
      }

      smallestFit = Integer.MAX_VALUE;

      //	Look for ranges that have cardinality of a multiple of the number of roles
      for (int i = 0; i < fnInfo.paramRanges.size(); i++)
      {
        int numValues = fnInfo.paramRanges.get(i).size();

        if (numValues != ignoreSize &&
            numValues < smallestFit &&
            (numValues % numRoles == 0 || numValues % numRoles == 1))
        {
          smallestFit = numValues;
          smallestFitIndex = i;
        }
      }

      if (smallestFitIndex != -1 && smallestFitIndex <= MAX_NUM_PIECES)
      {
        potentialPiecePropSets
            .add(new PotentialPiecePropSet(fnInfo.getName(), smallestFitIndex));
      }
    }

    for (PotentialPiecePropSet pieceSet : potentialPiecePropSets)
    {
      GdlFunctionInfo info = basePropFns.get(pieceSet.fnName);

      for (String roleArgValue : info.paramRanges
          .get(pieceSet.potentialRoleArgIndex))
      {
        Set<GdlSentence> pieceSetSentences = new HashSet<>();

        for (GdlSentence baseProp : stateMachine.getBasePropositions())
        {
          if (baseProp.arity() == 1 &&
              baseProp.getBody().get(0) instanceof GdlFunction)
          {
            GdlFunction propFn = (GdlFunction)baseProp.getBody().get(0);

            if (propFn.arity() >= MIN_PIECE_PROP_ARITY &&
                propFn.arity() <= MAX_PIECE_PROP_ARITY &&
                pieceSet.fnName.equals(propFn.getName().toString()) &&
                propFn.getBody().get(pieceSet.potentialRoleArgIndex)
                    .toString().equals(roleArgValue))
            {
              pieceSetSentences.add(baseProp);
            }
          }
        }

        if (pieceSetSentences.size() >= MIN_PIECES_THRESHOLD)
        {
          LOGGER.debug("Possible piece set: " + pieceSetSentences);

          ForwardDeadReckonInternalMachineState pieceMask = stateMachine.createInternalState(
                                                                                   new MachineState(pieceSetSentences));
          PieceMaskSpecifier pieceSpecifier = new PieceMaskSpecifier(pieceMask);
          propGroupScoreSets.put(pieceSpecifier, new HeuristicScoreInfo(numRoles));
        }
      }
    }

    return propGroupScoreSets.size() > 0;
  }

  @Override
  public void tuningStartSampleGame()
  {
    for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets
        .entrySet())
    {
      HeuristicScoreInfo heuristicInfo = e.getValue();
      heuristicInfo.lastValue = -1;

      for(PieceUsageStats usageStats : e.getKey().individualPieceUsageStats)
      {
        usageStats.startSampleGame();
      }
    }
  }

  @Override
  public void tuningInterimStateSample(ForwardDeadReckonInternalMachineState sampleState,
                                       int choosingRoleIndex)
  {
    for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets
        .entrySet())
    {
      int currentValue = e.getKey().overallPieceMask.intersectionSize(sampleState);

      HeuristicScoreInfo heuristicInfo = e.getValue();
      if (heuristicInfo.lastValue != -1)
      {
        if (heuristicInfo.lastValue == currentValue)
        {
          heuristicInfo.noChangeTurnRate++;
        }
        else if (choosingRoleIndex >= 0)
        {
          heuristicInfo.hasRoleChanges[choosingRoleIndex] = true;
        }
        else if (choosingRoleIndex == -1)
        {
          //	In a simultaneous turn game all roles choose
          for (int i = 0; i < numRoles; i++)
          {
            heuristicInfo.hasRoleChanges[i] = true;
          }
        }
      }

      heuristicInfo.lastValue = currentValue;

      for(PieceUsageStats usageStats : e.getKey().individualPieceUsageStats)
      {
        usageStats.sampleMove(sampleState);
      }
    }

    totalSimulatedTurns++;
  }

  @Override
  public void tuningTerminalStateSample(ForwardDeadReckonInternalMachineState finalState,
                                        int[] roleScores)
  {
    for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets.entrySet())
    {
      int heuristicScore = finalState.intersectionSize(e.getKey().overallPieceMask);
      e.getValue().accrueSample(heuristicScore, roleScores);

      for(PieceUsageStats usageStats : e.getKey().individualPieceUsageStats)
      {
        usageStats.endSampleGame(roleScores);
      }
    }
  }

  @Override
  public void tuningComplete()
  {
    mTuningComplete = true;

    for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets.entrySet())
    {
      HeuristicScoreInfo heuristicInfo = e.getValue();

      heuristicInfo.noChangeTurnRate /= totalSimulatedTurns;
    }

    for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets.entrySet())
    {
      if (e.getValue().noChangeTurnRate < 0.5)
      {
        LOGGER.debug("Eliminating potential piece set with no-change rate: " +
                     e.getValue().noChangeTurnRate + ": " + e.getKey());
      }
      else
      {
        double[] roleCorrelations = e.getValue().getRoleCorrelations();

        LOGGER.debug("Correlations for piece set: " + e.getKey());
        for (int i = 0; i < numRoles; i++)
        {
          LOGGER.debug("  Role " + i + ": " + roleCorrelations[i]);

          if (roleCorrelations[i] >= MIN_HEURISTIC_CORRELATION)
          {
            if (!e.getValue().hasRoleChanges[i])
            {
              LOGGER.debug("Eliminating potential piece set with no role decision changes for correlated role: " +
                           e.getKey());
            }
            else
            {
              LOGGER.debug("Using piece set above for role above");
              if (pieceSets == null)
              {
                pieceSets = new PieceMaskSpecifier[numRoles];
              }

              if (pieceSets[i] == null)
              {
                pieceSets[i] = e.getKey();
              }
              else
              {
                pieceSets[i].merge(e.getKey());
              }
            }
          }
          else
          {
            LOGGER.debug("Piece set insufficiently correlated for role");
          }
        }
      }
    }

    //  If we found no positively correlated piece set then (for two player games only) look for negatively correlated ones
    //  and switch the sets over if found.  This is a fairly crude approach (that doesn't extend easily from 2-player games),
    //  but for now it's a smaller and easier to test change to what we have than the more general approach of allowing
    //  negated heuristic values (because they tend to change on the other player's turn)
    if ( pieceSets == null && numRoles == 2 )
    {
      for (Entry<PieceMaskSpecifier, HeuristicScoreInfo> e : propGroupScoreSets.entrySet())
      {
        if (e.getValue().noChangeTurnRate >= 0.5)
        {
          double[] roleCorrelations = e.getValue().getRoleCorrelations();

          for (int i = 0; i < numRoles; i++)
          {
            if (roleCorrelations[i] <= -MIN_HEURISTIC_CORRELATION)
            {
              //  opposite sense of test since we're planning to swap over the piece sets
              if (!e.getValue().hasRoleChanges[i])
              {
                if (pieceSets == null)
                {
                  pieceSets = new PieceMaskSpecifier[numRoles];
                }

                if (pieceSets[i] == null)
                {
                  pieceSets[i] = e.getKey();
                }
                else
                {
                  pieceSets[i].merge(e.getKey());
                }
              }
            }
          }
        }
      }

      //  Swap the piece sets over
      if ( pieceSets != null )
      {
        PieceMaskSpecifier temp = pieceSets[0];

        pieceSets[0] = pieceSets[1];
        pieceSets[1] = temp;
      }
    }

    // Check that all roles have a set of pieces.  If not, disable the heuristic.
    if (pieceSets != null)
    {
      LOGGER.debug("Some roles have piece sets");
      for (int i = 0; i < numRoles; i++)
      {
        LOGGER.debug("  Checking role " + i);
        if (pieceSets[i] == null)
        {
          LOGGER.debug("      No piece set for this role");
          LOGGER.debug("Piece heuristic only identified for a subset of roles - disabling");
          pieceSets = null;
          break;
        }
        LOGGER.debug("    Final piece set is: " + pieceSets[i]);
      }

      if (pieceSets != null)
      {
        LOGGER.debug("All roles have piece sets");
        for (int i = 0; i < numRoles; i++)
        {
          pieceSets[i].finalizePieceValues(i);

          LOGGER.info("Role " + i + " will use piece set " + pieceSets[i]);
        }
      }
    }
  }

  /**
   * Get the heuristic value for the specified state.
   *
   * @param xiState           - the state (never a terminal state).
   * @param xiPreviousState   - the previous state (can be null).
   * @param xiReferenceState  - state with which to compare to determine heuristic values
   */
  @Override
  public void getHeuristicValue(ForwardDeadReckonInternalMachineState xiState,
                                ForwardDeadReckonInternalMachineState xiPreviousState,
                                ForwardDeadReckonInternalMachineState xiReferenceState,
                                HeuristicInfo resultInfo)
  {
    //  If a piece value is 0 thn we risk divide by zero issues, so add EPSILON
    double ourPieceValue = pieceSets[0].getValue(xiState) + EPSILON;
    double theirPieceValue = pieceSets[1].getValue(xiState) + EPSILON;
    double proportion = (ourPieceValue - theirPieceValue) / (ourPieceValue + theirPieceValue);
    double referenceProportion = 0;

    if ( xiPreviousState != null )
    {
      double ourPreviousPieceValue = pieceSets[0].getValue(xiPreviousState) + EPSILON;
      double theirPreviousPieceValue = pieceSets[1].getValue(xiPreviousState) + EPSILON;
      double previousProportion = (ourPreviousPieceValue - theirPreviousPieceValue) / (ourPreviousPieceValue + theirPreviousPieceValue);

      resultInfo.treatAsSequenceStep = (Math.abs( proportion - previousProportion ) > MIN_HEURISTIC_DIFF_FOR_SEQUENCE);

      if ( xiReferenceState == xiPreviousState )
      {
        referenceProportion = (ourPreviousPieceValue - theirPreviousPieceValue) / (ourPreviousPieceValue + theirPreviousPieceValue);
      }
      else
      {
        double ourReferencePieceValue = pieceSets[0].getValue(xiReferenceState) + EPSILON;
        double theirReferencePieceValue = pieceSets[1].getValue(xiReferenceState) + EPSILON;

        referenceProportion = (ourReferencePieceValue - theirReferencePieceValue) / (ourReferencePieceValue + theirReferencePieceValue);
      }
    }
    else
    {
      proportion = 0;
      resultInfo.treatAsSequenceStep = false;
    }

    double sigmaScaling = (ourPieceValue+theirPieceValue)/4;

    resultInfo.heuristicValue[0] = 100 * sigma(sigmaScaling*(proportion-referenceProportion));
    resultInfo.heuristicValue[1] = 100 - resultInfo.heuristicValue[0];

    resultInfo.heuristicWeight = heuristicSampleWeight;
  }

  private static double sigma(double value)
  {
    return 1/(1 + Math.exp(-value));
  }

  @Override
  public void newTurn(ForwardDeadReckonInternalMachineState xiState, TreeNode xiNode)
  {
    rootNode = xiNode;

    if (pieceSets != null)
    {
      double total = 0;
      double ourPieceValue = 0;

      for (int i = 0; i < numRoles; i++)
      {
        rootPieceValues[i] = pieceSets[i].getValue(xiState);
        total += rootPieceValues[i];

        if (i == 0)
        {
          ourPieceValue = total;
        }
      }

      double averageMaterial = total/numRoles;
      double ourMaterialDivergence = ourPieceValue - averageMaterial;

      //  Weight further material gain down the more we're already ahead/behind in material
      //  because in either circumstance it's likely to be position that is more important
      heuristicSampleWeight = (int)Math.max(2, 6.5 - Math.abs(ourMaterialDivergence)*48/averageMaterial);

      LOGGER.info("Piece heuristic weight set to: " + heuristicSampleWeight);
    }
    else
    {
      heuristicSampleWeight = 0;
    }
  }

  @Override
  public boolean isEnabled()
  {
    // We're enabled by default, unless we fail to discover piece sets by the end of tuning.
    return ((!mTuningComplete) || (pieceSets != null));
  }

  @Override
  public Heuristic createIndependentInstance()
  {
    return new PieceHeuristic(this);
  }
}
