package org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.player.gamer.statemachine.sancho.RuntimeGameCharacteristics;
import org.ggp.base.util.gdl.grammar.GdlConstant;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.propnet.polymorphic.PolymorphicComponent;
import org.ggp.base.util.propnet.polymorphic.PolymorphicOr;
import org.ggp.base.util.propnet.polymorphic.PolymorphicProposition;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonProposition;

/**
 * Class responsible for analysing a game's propnet to determine its control set and factors.
 */
public class FactorAnalyser
{
  private static final Logger LOGGER = LogManager.getLogger();

  private ForwardDeadReckonPropnetStateMachine mStateMachine;

  //  We don't accept more than a reasonable smallish number of factors since
  //  1) The way we instantiate search trees for each only scales modestly
  //  2) A very large apparent number of factors is probably an artifact of the fact that we don't currently
  //     handle goal couplings properly in factor detection, which has been seen to result
  //     in spurious factorization in 'Guess Two Thirds'
  static final private int            MAX_FACTORS = 16;

  private int                                             mNumBaseProps;
  private final Collection<PolymorphicProposition>        mBasePropositions;
  private final Map<PolymorphicComponent, DependencyInfo> mBasePropositionDependencies = new HashMap<>();
  private final DependencyCache                           mComponentDirectBaseDependencies;

  private class DirectDependencyInfo
  {
    public Set<PolymorphicProposition>  dependencies = new HashSet<>();
    public Set<ForwardDeadReckonLegalMoveInfo> moves = new HashSet<>();

    public DirectDependencyInfo()
    {
    }

    public void add(DirectDependencyInfo other)
    {
      dependencies.addAll(other.dependencies);
      moves.addAll(other.moves);
    }

    public void buildImmediateBaseDependencies(PolymorphicComponent c, Set<PolymorphicComponent> pathSet)
    {
      FactorAnalyser.this.buildImmediateBaseDependencies(c, this, pathSet);
    }
  }

  // An LRU cache is used to cache component dependencies.  The size of this cache is a trade-off between performance of
  // walking the dependency graph and memory consumption (which can equate to performance via GC time).
  private class DependencyInfo extends DirectDependencyInfo
  {
    public DependencyInfo()
    {
    }

    @Override
    public void buildImmediateBaseDependencies(PolymorphicComponent c, Set<PolymorphicComponent> pathSet)
    {
      FactorAnalyser.this.buildImmediateBaseDependencies(c, this, pathSet);

      //  The 0th level dependencies are the immediate dependencies
      dependenciesByLevel.add(new HashSet<>(dependencies));
      movesByLevel.add(new HashSet<>(moves));
    }

    public List<Set<PolymorphicProposition>>  dependenciesByLevel = new ArrayList<>();
    public List<Set<ForwardDeadReckonLegalMoveInfo>>  movesByLevel = new ArrayList<>();
  }

  private class DependencyCache extends LinkedHashMap<PolymorphicComponent, DirectDependencyInfo>
  {
    private static final long serialVersionUID = 1L;
    private int               maxEntries;

    public DependencyCache(int capacity)
    {
      super(capacity + 1, 1.0f, true);
      maxEntries = capacity;
    }

    @Override
    protected boolean removeEldestEntry(final Map.Entry<PolymorphicComponent, DirectDependencyInfo> eldest)
    {
      return super.size() > maxEntries;
    }
  }

  public FactorAnalyser(ForwardDeadReckonPropnetStateMachine stateMachine)
  {
    this.mStateMachine = stateMachine;
    mBasePropositions = stateMachine.getFullPropNet().getBasePropositions().values();
    mNumBaseProps = mBasePropositions.size();
    mComponentDirectBaseDependencies = new DependencyCache(5000);
  }

  private DirectDependencyInfo getComponentDirectDependencies(PolymorphicComponent c, Set<PolymorphicComponent> pathSet)
  {
    if (mBasePropositions.contains(c))
    {
      if (!mBasePropositionDependencies.containsKey(c))
      {
        DependencyInfo newInfo = new DependencyInfo();

        //  Do the put into the map BEFORE trying to recurse as otherwise loops
        //  will be infinite!
        mBasePropositionDependencies.put(c, newInfo);
        newInfo.buildImmediateBaseDependencies(c, pathSet);

        return newInfo;
      }

      return mBasePropositionDependencies.get(c);
    }
    else if (!mComponentDirectBaseDependencies.containsKey(c))
    {
      DirectDependencyInfo newInfo = new DirectDependencyInfo();

      //  Do the put into the map BEFORE trying to recurse as otherwise loops
      //  will be infinite!  We also cache components with many inputs, as they will
      //  be expensive to recalculate
      if (c instanceof PolymorphicProposition || c.getInputs().size() > 5)
      {
        mComponentDirectBaseDependencies.put(c, newInfo);
      }
      newInfo.buildImmediateBaseDependencies(c, pathSet);

      return newInfo;
    }

    return mComponentDirectBaseDependencies.get(c);
  }

  /**
   * Identify the control set and any factors.
   *
   * @param xiTimeout             - When to give up if the analysis takes too long.
   * @param xiGameCharacteristics - Information learned about this game.
   *
   * @return the factor information.
   */
  public FactorInfo run(long xiTimeout, RuntimeGameCharacteristics xiGameCharacteristics)
  {
    // Attempt to reload the analysis.
    boolean lReloaded = true;
    FactorInfo lFactorInfo = reloadAnalysis(xiGameCharacteristics);
    if ((!lFactorInfo.mControlSetCalculated) || (!lFactorInfo.mFactorsCalculated))
    {
      lReloaded = false;

      // Missing some information.  Perform the analysis - if we've got more time than before.
      if (xiTimeout > xiGameCharacteristics.getMaxFactorFailureTime() * 1.25)
      {
        LOGGER.info("Performing factor analysis");
        boolean lAlreadyKnewControlSet = lFactorInfo.mControlSetCalculated;
        analyse(lFactorInfo, xiTimeout, xiGameCharacteristics);

        // If we've just learned the control set for the first time, record it.
        if ((lFactorInfo.mControlSetCalculated) && (!lAlreadyKnewControlSet))
        {
          // Build a string representation.  We can't just do lFactorInfo.mControlSet.toString() because it contains
          // spurious detail.
          StringBuilder lControlSet = new StringBuilder("( ");
          for (PolymorphicProposition lProp : lFactorInfo.mControlSet)
          {
            lControlSet.append(lProp.getName());
            lControlSet.append(", ");
          }
          if (lControlSet.length() > 2)
          {
            // Strip the trailing ", ".
            lControlSet.setLength(lControlSet.length() - 2);
          }
          lControlSet.append(" )");

          xiGameCharacteristics.setControlMask(lControlSet.toString());
        }
      }
      else
      {
        LOGGER.info("Not attempting to factor this game.  Previous attempt showed " +
                    xiGameCharacteristics.getNumFactors() + " factor(s) in " +
                    xiGameCharacteristics.getMaxFactorFailureTime() + "ms.");
      }
    }

    // Record the factor that each proposition belongs to.
    Set<Factor> lFactors = lFactorInfo.mFactors;
    if (lFactors != null && lFactors.size() > 1 && lFactors.size() <= MAX_FACTORS)
    {
      for (Factor lFactor : lFactors)
      {
        lFactor.dump();

        for (PolymorphicComponent c : lFactor.getComponents())
        {
          if (c instanceof PolymorphicProposition)
          {
            PolymorphicProposition p = (PolymorphicProposition)c;

            ForwardDeadReckonProposition fdrp = (ForwardDeadReckonProposition)p;
            ForwardDeadReckonPropositionCrossReferenceInfo lInfo = (ForwardDeadReckonPropositionCrossReferenceInfo)fdrp.getInfo();

            lInfo.factor = lFactor;
          }
        }
      }

      // Mark all the factors complete.
      for (Factor lFactor : lFactors)
      {
        lFactor.complete(mStateMachine);
      }

      // If this isn't just a copy of the reloaded state, be sure to save it.
      if (!lReloaded)
      {
        xiGameCharacteristics.setFactors(lFactors);
      }
    }

    // If we've learned for sure that there's only 1 factor, record that.
    if (lFactorInfo.mFactorsCalculated && lFactors == null)
    {
      Factor lDummyFactor = new Factor();
      Set<Factor> lDummyFactors = new HashSet<>();
      lDummyFactors.add(lDummyFactor);
      xiGameCharacteristics.setFactors(lDummyFactors);
    }

    return lFactorInfo;
  }

  /**
   * Identify the control set and any factors.
   *
   * @param xiFactorInfo          - Structure in which to store the results.
   * @param xiTimeout             - When to give up if the analysis takes too long.
   * @param xiGameCharacteristics - Information learned about this game.
   */
  public void analyse(FactorInfo xiFactorInfo, long xiTimeout, RuntimeGameCharacteristics xiGameCharacteristics)
  {
    // Perform a fresh analysis.  (This can take a while.)
    Set<PolymorphicProposition> lControlSet = new HashSet<>();
    Set<Factor> lFactors = new HashSet<>();
    long lAbortTime = System.currentTimeMillis() + xiTimeout;

    // A path for the current recursion through the network is always maintained to allow loop detection.
    Set<PolymorphicComponent> pathSet = new HashSet<>();

    //  Construct the full closure of base dependencies
    for (PolymorphicProposition baseProp : mBasePropositions)
    {
      pathSet.clear();
      DependencyInfo dInfo = (DependencyInfo)getComponentDirectDependencies(baseProp, pathSet);
      int depth = 1;

      Set<PolymorphicProposition> dependenciesAtDepth;
      Set<ForwardDeadReckonLegalMoveInfo> movesAtDepth;
      do
      {
        dependenciesAtDepth = new HashSet<>();
        dInfo.dependenciesByLevel.add(dependenciesAtDepth);
        movesAtDepth = new HashSet<>();
        dInfo.movesByLevel.add(movesAtDepth);

        for (PolymorphicProposition fringeDependency : dInfo.dependenciesByLevel.get(depth - 1))
        {
          pathSet.clear();
          DirectDependencyInfo ddInfo = getComponentDirectDependencies(fringeDependency, pathSet);
          dependenciesAtDepth.addAll(ddInfo.dependencies);
          movesAtDepth.addAll(ddInfo.moves);

          if (dependenciesAtDepth.size() >= mNumBaseProps)
          {
            break;
          }

          //  If the analysis is just taking, too long give up.
          if (System.currentTimeMillis() > lAbortTime)
          {
            LOGGER.warn("Factorization analysis timed out after at least " + xiTimeout + "ms");
            xiGameCharacteristics.factoringFailedAfter(xiTimeout);
            return;
          }
        }

        dependenciesAtDepth.removeAll(dInfo.dependencies);
        movesAtDepth.removeAll(dInfo.moves);

        dInfo.dependencies.addAll(dependenciesAtDepth);
        dInfo.moves.addAll(movesAtDepth);

        depth++;
      } while (!dependenciesAtDepth.isEmpty() && dInfo.dependencies.size() < mNumBaseProps);

      // If we find a base prop that depends on more than 2/3rds of the others assume we're not going to be able to
      // factorize, so minimize wasting time on the factorization analysis.  We cannot simply quit now however, since we
      // need to complete analysis of the control set if possible.
      if (dInfo.dependencies.size() > (mNumBaseProps * 2) / 3)
      {
        xiFactorInfo.mFactorsCalculated = true;
        lFactors = null;
      }
    }

    //  Now look for pure disjunctive inputs to goal and terminal
    Map<PolymorphicComponent, DependencyInfo> disjunctiveInputs = new HashMap<>();

    pathSet.clear();
    addDisjunctiveInputProps(mStateMachine.getFullPropNet().getTerminalProposition(), disjunctiveInputs, pathSet);
    //  TODO - same for goals

    //  Trim out from each disjunctive input set those propositions in the control set, which are only
    //  influenced by other base props independently of moves (usually step and control logic)
    Set<PolymorphicComponent> controlOnlyInputs = new HashSet<>();

    for (PolymorphicProposition baseProp : mBasePropositions)
    {
      pathSet.clear();
      DependencyInfo dInfo = (DependencyInfo)getComponentDirectDependencies(baseProp, pathSet);

      if (dInfo.moves.isEmpty())
      {
        lControlSet.add(baseProp);
      }
    }

    xiFactorInfo.mControlSetCalculated = true;
    xiFactorInfo.mControlSet = lControlSet;

    LOGGER.info("Control set: " + lControlSet);
    if (lFactors == null)
    {
      return;
    }

    for (Entry<PolymorphicComponent, DependencyInfo> e : disjunctiveInputs.entrySet())
    {
      e.getValue().dependencies.removeAll(lControlSet);
      if (e.getValue().dependencies.isEmpty())
      {
        controlOnlyInputs.add(e.getKey());
      }
    }

    //  Trim out those inputs which have no base dependencies
    for (PolymorphicComponent c : controlOnlyInputs)
    {
      disjunctiveInputs.remove(c);
    }

    //  TOTAL HACK - TEMPORARY (I hope)
    //  The way the GDL is written for several simultaneous games introduces an
    //  artificial pseudo-coupling for the termination condition that amounts to
    //  someone-has-no-moves-left 'stalemates'.  Examples are:
    //    All board positions occupied in C4
    //    One player has no pieces left in Breakthrough
    //  The GDL tends to express these as conditions across both games in factorized
    //  situations.  For instance simultaneous C4 expresses this as 'there are no blank
    //  cells left on either board' which amounts to a NOT of a large OR for every cell
    //  being red of black.
    //  In general these psuedo-couplings are quite hard to pick out in static analysis
    //  and there are a number of possible (somewhat) robust approaches we could try:
    //    1)  Disconnect them from the terminal clause and directly monitor them during
    //        simulation, but replace them in the terminal disjunction with the extra
    //        pseudo-condition 'the legal move set is empty'.  During metagaming simulate
    //        this modified game, and if the suspect disjunct we are monitoring directly
    //        only goes TRUE in states that the modified game finds to ANYWAY be terminal
    //        then they are not true couplings of the factors.
    //    2)  Rewrite them as a disjunctive approximation by making them ORs of two
    //        instances of their input logic obtained by pruning out all base props
    //        from one of the factors.  Simulate both versions of the game, and if
    //        they never disagree about termination then the coupling is not a true
    //        coupling and we can use the factored approximation
    //  HOWEVER, for now we take a much cheaper seem-to-get-away-with-it approach
    //  and consider the size of the dependency sets for the disjunctive propositions - if
    //  any of them involve more than half of the (non-control) base-props we assume they
    //  can be ignored for the purposes of determining factorization!
    Set<PolymorphicComponent> ignorableDisjuncts = new HashSet<>();

    for (Entry<PolymorphicComponent, DependencyInfo> e : disjunctiveInputs.entrySet())
    {
      if (e.getValue().dependencies.size() > (mNumBaseProps - lControlSet.size()) / 2)
      {
        ignorableDisjuncts.add(e.getKey());
      }
    }

    for (PolymorphicComponent c : ignorableDisjuncts)
    {
      disjunctiveInputs.remove(c);
    }

    // Now find sets of disjunctive inputs with non-intersecting base prop dependencies - these are the factors.
    while (!disjunctiveInputs.isEmpty())
    {
      //  If the analysis is just taking too long give up
      if (System.currentTimeMillis() > lAbortTime)
      {
        LOGGER.warn("Factorization analysis (post dependency phase) timed out after at least " + xiTimeout + "ms");
        xiGameCharacteristics.factoringFailedAfter(xiTimeout);
        return;
      }

      Factor lNewFactor = new Factor();

      lNewFactor.addAll(disjunctiveInputs.values().iterator().next().dependencies);
      lFactors.add(lNewFactor);

      for (Factor lFactor : lFactors)
      {
        boolean anyAdded;

        do
        {
          anyAdded = false;

          Set<PolymorphicComponent> inputsProcessed = new HashSet<>();

          for (Entry<PolymorphicComponent, DependencyInfo> e : disjunctiveInputs.entrySet())
          {
            if (lFactor.containsAny(e.getValue().dependencies))
            {
              lFactor.addAll(e.getValue().dependencies);
              lFactor.addAllMoves(e.getValue().moves);
              inputsProcessed.add(e.getKey());
              anyAdded = true;
            }
          }

          for (PolymorphicComponent p : inputsProcessed)
          {
            disjunctiveInputs.remove(p);
          }
        } while (anyAdded);
      }
    }

    // Factor analysis is complete.
    xiFactorInfo.mFactorsCalculated = true;
    if (lFactors.size() > 1 && lFactors.size() <= MAX_FACTORS)
    {
      xiFactorInfo.mFactors = lFactors;
    }

    return;
  }

  private void addDisjunctiveInputProps(PolymorphicComponent c, Map<PolymorphicComponent, DependencyInfo> disjunctiveInputs, Set<PolymorphicComponent> pathSet)
  {
    recursiveAddDisjunctiveInputProps(c.getSingleInput(), disjunctiveInputs, pathSet);
  }

  private void recursiveAddDisjunctiveInputProps(PolymorphicComponent c, Map<PolymorphicComponent, DependencyInfo> disjunctiveInputs, Set<PolymorphicComponent> pathSet)
  {
    if (c instanceof PolymorphicOr)
    {
      for (PolymorphicComponent input : c.getInputs())
      {
        recursiveAddDisjunctiveInputProps(input, disjunctiveInputs, pathSet);
      }
    }
    else
    {
      //  For each disjunctive input find the base props it is dependent on
      DirectDependencyInfo immediateDependencies = getComponentDirectDependencies(c, pathSet);

      DependencyInfo fullDependencies = new DependencyInfo();

      //  Construct the full dependencies from the already calculated base dependencies map
      for (PolymorphicProposition p : immediateDependencies.dependencies)
      {
        DirectDependencyInfo immediateDependentDependencies = getComponentDirectDependencies(p, pathSet);

        fullDependencies.add(immediateDependentDependencies);
        if (fullDependencies.dependencies.size() >= mNumBaseProps)
        {
          break;
        }
      }
      disjunctiveInputs.put(c, fullDependencies);
    }
  }

  DirectDependencyInfo buildImmediateBaseDependencies(PolymorphicComponent xiC, DirectDependencyInfo result, Set<PolymorphicComponent> pathSet)
  {
    recursiveBuildImmediateBaseDependencies(xiC, xiC, result, pathSet);

    return result;
  }

  private void recursiveBuildImmediateBaseDependencies(PolymorphicComponent root, PolymorphicComponent c, DirectDependencyInfo dInfo, Set<PolymorphicComponent> pathSet)
  {
    if (pathSet.contains(c))
    {
      return;
    }
    pathSet.add(c);

    try
    {
      if (c instanceof PolymorphicProposition)
      {
        if (dInfo.dependencies.contains(c))
        {
          return;
        }

        PolymorphicProposition p = (PolymorphicProposition)c;
        GdlConstant name = p.getName().getName();

        if (mBasePropositions.contains(p))
        {
          dInfo.dependencies.add(p);
          if (root != p)
          {
            pathSet.remove(c);
            return;
          }
        }

        if (name.equals(GdlPool.INIT))
        {
          return;
        }

        if (name.equals(GdlPool.DOES))
        {
          if (root != null)
          {
            PolymorphicProposition legalProp = mStateMachine.getFullPropNet().getLegalInputMap().get(c);

            if (legalProp != null)
            {
              DirectDependencyInfo legalPropInfo = getComponentDirectDependencies(legalProp, pathSet);

              ForwardDeadReckonLegalMoveInfo[] masterMoveList = mStateMachine.getFullPropNet().getMasterMoveList();
              ForwardDeadReckonLegalMoveInfo moveInfo = masterMoveList[((ForwardDeadReckonProposition)legalProp).getInfo().index];

              dInfo.add(legalPropInfo);

              dInfo.moves.add(moveInfo);
            }
          }

          return;
        }
      }

      //  The following is a somewhat heuristic approach to getting around 'fake' dependencies
      //  introduced by distinct clauses in the GDL.  Specifically games often have rules of this
      //  type (taken from KnighThrough):
      //    (<= (next (cell ?x ?y ?state))
      //        (true (cell ?x ?y ?state))
      //        (does ?player (move ?x1 ?y1 ?x2 ?y2))
      //        (distinctcell ?x ?y ?x1 ?y1)
      //        (distinctcell ?x ?y ?x2 ?y2))
      //  This is saying that a cell retains its state unless someone moves into or out of it
      //  The problem is that it encodes this logic by testing that what is played is one of the
      //  set of moves that is not a move into or out of the cell in question - i.e. - a vast OR
      //  consisting of most moves in the game, and in particular, including all the moves on the
      //  'other board' relative to the tested cell.  Naively this coupling of a move on the other
      //  board to the next state of a cell on this board looks like a dependency that prevents
      //  factorization, but we must prevent this being seen as a 'real' coupling.  This is a result
      //  of the (not encoded into the propnet) requirement (for a well formed game) that exactly
      //  one move must be played each turn, so this huge OR is equivalent to the NOT of a much smaller
      //  OR for the complimentary set of moves.  We convert ORs involving more than half of the moves in
      //  the game with that transformation and calculate the dependencies with respect to the result.
      if (c instanceof PolymorphicOr)
      {
        Collection<PolymorphicProposition> inputProps = mStateMachine.getFullPropNet().getInputPropositions().values();
        if (c.getInputs().size() > inputProps.size() / 2)
        {
          Set<PolymorphicComponent> inputPropsOred = new HashSet<>();

          for (PolymorphicComponent input : c.getInputs())
          {
            if (inputProps.contains(input))
            {
              inputPropsOred.add(input);
            }
          }

          if (inputPropsOred.size() > inputProps.size() / 2)
          {
            for (PolymorphicComponent input : c.getInputs())
            {
              if (!inputPropsOred.contains(input))
              {
                DirectDependencyInfo inputPropInfo = getComponentDirectDependencies(input, pathSet);

                dInfo.add(inputPropInfo);
              }
            }
            for (PolymorphicComponent input : inputProps)
            {
              if (!inputPropsOred.contains(input))
              {
                DirectDependencyInfo inputPropInfo = getComponentDirectDependencies(input, pathSet);

                dInfo.add(inputPropInfo);
              }
            }

            return;
          }
        }
      }

      for (PolymorphicComponent input : c.getInputs())
      {
        if (mBasePropositions.contains(input))
        {
          dInfo.dependencies.add((PolymorphicProposition)input);
        }
        else
        {
          DirectDependencyInfo inputPropInfo = getComponentDirectDependencies(input, pathSet);

          dInfo.add(inputPropInfo);
        }
      }
    }
    finally
    {
      pathSet.remove(c);
    }
  }

  /**
   * Reload analysis from the game characteristics.
   *
   * @param xiCharacteristics - the characteristics.
   *
   * @return the factor information.
   */
  private FactorInfo reloadAnalysis(RuntimeGameCharacteristics xiCharacteristics)
  {
    FactorInfo lFactorInfo = new FactorInfo();

    // Load the control mask.
    String lSavedControlMask = xiCharacteristics.getControlMask();
    if (lSavedControlMask != null)
    {
      lFactorInfo.mControlSet = reloadControlSet(lSavedControlMask);
      lFactorInfo.mControlSetCalculated = true;
    }

    // Load the factors.
    int lNumFactors = xiCharacteristics.getNumFactors();
    if (lNumFactors > 0)
    {
      lFactorInfo.mFactorsCalculated = true;

      if (lNumFactors > 1)
      {
        String lSavedFactors = xiCharacteristics.getFactors();
        if (lSavedFactors == null)
        {
          // For games saved before we saved factors, we might know how many there are without knowing what they are.
          // Calculate them again from scratch.
          lFactorInfo.mFactorsCalculated = false;
        }
        else
        {
          lFactorInfo.mFactors = reloadFactors(lSavedFactors);
        }
      }
    }

    return lFactorInfo;
  }

  private Set<PolymorphicProposition> reloadControlSet(String xiSavedState)
  {
    LOGGER.debug("Reloading control set: " + xiSavedState);

    Set<PolymorphicProposition> lControlSet = new HashSet<>();

    // Strip the surrounding "( " + " )"
    xiSavedState = xiSavedState.substring(2, xiSavedState.length() - 2);

    if ( !xiSavedState.isEmpty())
    {
      // Split on ", "
      String[] lProps = xiSavedState.split(", ");
      SET_PROPS: for (String lProp : lProps)
      {
        // Find the matching prop in the info set.
        for (ForwardDeadReckonPropositionCrossReferenceInfo lInfo : mStateMachine.getInfoSet())
        {
          if (lInfo.fullNetProp.getName().toString().equals(lProp))
          {
            lControlSet.add(lInfo.fullNetProp);
            continue SET_PROPS;
          }
        }
        assert(false) : "Failed to find proposition: " + lProp;
      }
    }

    return lControlSet;
  }

  private Set<Factor> reloadFactors(String xiSavedState)
  {
    LOGGER.debug("Reloading factors: " + xiSavedState);

    Set<Factor> lFactors = new HashSet<>();
    for (String lFactor : xiSavedState.split(";"))
    {
      lFactors.add(new Factor(lFactor, mStateMachine.getInfoSet(), mStateMachine.getMasterLegalMoves()));
    }

    return lFactors;
  }

  /**
   * Information about factors.
   */
  public static class FactorInfo
  {
    /**
     * Whether the control set has been calculated.
     */
    public boolean                     mControlSetCalculated = false;

    /**
     * The control set - i.e. those propositions which can't ever affect the goals (e.g. step counters).
     */
    public Set<PolymorphicProposition> mControlSet = new HashSet<>();

    /**
     * Whether the factors have been calculated.
     */
    public boolean                     mFactorsCalculated = false;

    /**
     * The factors.  If the factors have been calculated, but this is null, the game is known not to split into factors.
     */
    public Set<Factor>                 mFactors;
  }
}
