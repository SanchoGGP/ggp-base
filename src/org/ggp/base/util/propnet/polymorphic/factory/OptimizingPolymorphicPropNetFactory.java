package org.ggp.base.util.propnet.polymorphic.factory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.util.concurrency.ConcurrencyUtils;
import org.ggp.base.util.gdl.GdlUtils;
import org.ggp.base.util.gdl.grammar.Gdl;
import org.ggp.base.util.gdl.grammar.GdlConstant;
import org.ggp.base.util.gdl.grammar.GdlDistinct;
import org.ggp.base.util.gdl.grammar.GdlLiteral;
import org.ggp.base.util.gdl.grammar.GdlNot;
import org.ggp.base.util.gdl.grammar.GdlPool;
import org.ggp.base.util.gdl.grammar.GdlProposition;
import org.ggp.base.util.gdl.grammar.GdlRelation;
import org.ggp.base.util.gdl.grammar.GdlRule;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.gdl.grammar.GdlVariable;
import org.ggp.base.util.gdl.model.SentenceDomainModel;
import org.ggp.base.util.gdl.model.SentenceDomainModelFactory;
import org.ggp.base.util.gdl.model.SentenceDomainModelOptimizer;
import org.ggp.base.util.gdl.model.SentenceForm;
import org.ggp.base.util.gdl.model.SentenceForms;
import org.ggp.base.util.gdl.model.SentenceModelUtils;
import org.ggp.base.util.gdl.model.assignments.AssignmentIterator;
import org.ggp.base.util.gdl.model.assignments.Assignments;
import org.ggp.base.util.gdl.model.assignments.AssignmentsFactory;
import org.ggp.base.util.gdl.model.assignments.FunctionInfo;
import org.ggp.base.util.gdl.model.assignments.FunctionInfoImpl;
import org.ggp.base.util.gdl.transforms.CommonTransforms;
import org.ggp.base.util.gdl.transforms.CondensationIsolator;
import org.ggp.base.util.gdl.transforms.ConstantChecker;
import org.ggp.base.util.gdl.transforms.ConstantCheckerFactory;
import org.ggp.base.util.gdl.transforms.DeORer;
import org.ggp.base.util.gdl.transforms.GdlCleaner;
import org.ggp.base.util.gdl.transforms.Relationizer;
import org.ggp.base.util.gdl.transforms.VariableConstrainer;
import org.ggp.base.util.propnet.architecture.Component;
import org.ggp.base.util.propnet.architecture.PropNet;
import org.ggp.base.util.propnet.architecture.components.Proposition;
import org.ggp.base.util.propnet.polymorphic.PolymorphicAnd;
import org.ggp.base.util.propnet.polymorphic.PolymorphicComponent;
import org.ggp.base.util.propnet.polymorphic.PolymorphicComponentFactory;
import org.ggp.base.util.propnet.polymorphic.PolymorphicConstant;
import org.ggp.base.util.propnet.polymorphic.PolymorphicNot;
import org.ggp.base.util.propnet.polymorphic.PolymorphicOr;
import org.ggp.base.util.propnet.polymorphic.PolymorphicPropNet;
import org.ggp.base.util.propnet.polymorphic.PolymorphicProposition;
import org.ggp.base.util.propnet.polymorphic.PolymorphicTransition;
import org.ggp.base.util.statemachine.Role;

import com.google.common.collect.Multimap;

/**
 * A propnet factory meant to optimize the propnet before it's even built, mostly through transforming the GDL.
 * (The transformations identify certain classes of rules that have poor performance and replace them with equivalent
 * rules that have better performance, with performance measured by the size of the propnet.)
 *
 * Known issues:
 *
 * <ol>
 *
 * <li> Does not work on games with many advanced forms of recursion. These include:<ol>
 *   <li> Anything that breaks the SentenceModel
 *   <li> Multiple sentence forms which reference one another in rules
 *   <li> Not 100% confirmed to work on games where recursive rules have multiple recursive conjuncts</ol>
 *
 * <li> Currently runs some of the transformations multiple times. A Description object containing information about the
 *      description and its properties would alleviate this.
 *
 * <li> Its current solution to the "unaffected piece rule" problem is somewhat clumsy and ungeneralized, relying on the
 *      combined behaviors of CrudeSplitter and CondensationIsolator.
 *
 * <li> The mutex finder in particular is very ungeneralized. It should be replaced with a more general mutex finder.
 *
 * <li> Actually, the referenced solution is not even enabled at the moment. It may not be working even with the proper
 *      options set.
 *
 * <li> Depending on the settings and the situation, the behavior of the CondensationIsolator can be either too
 *      aggressive or not aggressive enough.  Both result in excessively large games.  A more sophisticated version of
 *      the CondensationIsolator could solve these problems.  A stopgap alternative is to try both settings and use the
 *      smaller propnet (or the first to be created, if multithreading).
 *
 * Adapted from OptimizingPropNetFactory as supplied upstream.
 */
public class OptimizingPolymorphicPropNetFactory
{
  private static final Logger LOGGER = LogManager.getLogger();

  //TODO: This currently doesn't actually give a different constant from INIT
  static final private GdlConstant    INIT_CAPS = GdlPool.getConstant("INIT");
  static final private GdlProposition TEMP      = GdlPool.getProposition(GdlPool.getConstant("TEMP"));

  /**
   * @return a PropNet for the game with the given description.
   *
   * @param xiDescription      - the GDL description of the game.
   * @param xiComponentFactory - a factory for creating individual propnet components.
   *
   * @throws InterruptedException
   *           if the thread is interrupted during PropNet creation.
   */
  public static PolymorphicPropNet create(List<Gdl> xiDescription,
                                          PolymorphicComponentFactory xiComponentFactory)
      throws InterruptedException
  {
    LOGGER.debug("Building propnet");

    // Perform transformations of the GDL.  Note that some of the later transformations re-run the earlier
    // transformations.  This is deliberate and is required because some later simplifications allow the earlier
    // simplifications to be applied again.
    xiDescription = GdlCleaner.run(xiDescription);
    xiDescription = DeORer.run(xiDescription);
    xiDescription = VariableConstrainer.replaceFunctionValuedVariables(xiDescription);
    xiDescription = Relationizer.run(xiDescription);
    xiDescription = CondensationIsolator.run(xiDescription);

    // Trace out the final GDL.
    for (Gdl gdl : xiDescription)
    {
      LOGGER.trace(gdl.getClass().getSimpleName() + ": " + gdl);
    }

    // We want to start with a rule graph and follow the rule graph.  Start by finding general information about the
    // game.
    SentenceDomainModel model = SentenceDomainModelFactory.createWithCartesianDomains(xiDescription);

    // Restrict domains to values that could actually come up in rules.  See the "toCount" relation in
    // base.chinesecheckers4 for an example of why this could be useful.  When used in the body of rules as
    // {@code (toCount ?player ?x ?y ?z)}, this relation has 4 * 37 * 37 * 37 (i.e. 202,612) possible sets of arguments.
    // However, of those, only 4 can ever be true!
    model = SentenceDomainModelOptimizer.restrictDomainsToUsefulValues(model);

    LOGGER.trace("Setting constants...");

    ConstantChecker constantChecker = ConstantCheckerFactory.createWithForwardChaining(model);
    LOGGER.trace("Done setting constants");

    Set<String> sentenceFormNames = SentenceForms.getNames(model.getSentenceForms());
    boolean usingBase = sentenceFormNames.contains("base");
    boolean usingInput = sentenceFormNames.contains("input");


    // For now, we're going to build this to work on those with a particular restriction on the dependency graph:
    //   Recursive loops may only contain one sentence form.
    //
    // This describes most games, but not all legal games.
    Multimap<SentenceForm, SentenceForm> dependencyGraph = model.getDependencyGraph();
    LOGGER.trace("Computing topological ordering... ");
    ConcurrencyUtils.checkForInterruption();
    List<SentenceForm> topologicalOrdering = getTopologicalOrdering(model.getSentenceForms(),
                                                                    dependencyGraph,
                                                                    usingBase,
                                                                    usingInput);
    LOGGER.trace("Done computing topological ordering");

    Role[] roles = Role.computeRoles(xiDescription);
    Map<GdlSentence, PolymorphicComponent> components = new HashMap<>();
    Map<GdlSentence, PolymorphicComponent> negations = new HashMap<>();
    PolymorphicConstant trueComponent = xiComponentFactory.createConstant(-1, true);
    PolymorphicConstant falseComponent = xiComponentFactory.createConstant(-1, false);
    Map<SentenceForm, FunctionInfo> functionInfoMap = new HashMap<>();
    Map<SentenceForm, Collection<GdlSentence>> completedSentenceFormValues = new HashMap<>();
    for (SentenceForm form : topologicalOrdering)
    {
      ConcurrencyUtils.checkForInterruption();

      LOGGER.trace("Adding sentence form: " + form);

      if (constantChecker.isConstantForm(form))
      {
        // We only add sentence in constant form if they are important.
        if (form.getName().equals(GdlPool.LEGAL) ||
            form.getName().equals(GdlPool.GOAL) ||
            form.getName().equals(GdlPool.INIT) ||
            form.getName().equals(GdlPool.NEXT) ||
            form.getName().equals(GdlPool.TERMINAL))
        {
          for (GdlSentence trueSentence : constantChecker.getTrueSentences(form))
          {
            // Create the proposition and wire it up to the 'true' constant.
            PolymorphicProposition trueProp = xiComponentFactory.createProposition(-1, trueSentence);
            trueProp.addInput(trueComponent);
            trueComponent.addOutput(trueProp);
            components.put(trueSentence, trueComponent);
          }
        }

        addConstantsToFunctionInfo(form, constantChecker, functionInfoMap);
        addFormToCompletedValues(form,
                                 completedSentenceFormValues,
                                 constantChecker);
      }
      else
      {
        //TODO: Adjust "recursive forms" appropriately
        //Add a temporary sentence form thingy? ...
        Map<GdlSentence, PolymorphicComponent> temporaryComponents = new HashMap<>();
        Map<GdlSentence, PolymorphicComponent> temporaryNegations = new HashMap<>();
        addSentenceForm(form,
                        model,
                        components,
                        negations,
                        trueComponent,
                        falseComponent,
                        usingBase,
                        usingInput,
                        Collections.singleton(form),
                        temporaryComponents,
                        temporaryNegations,
                        functionInfoMap,
                        constantChecker,
                        completedSentenceFormValues,
                        xiComponentFactory);
        //TODO: Pass these over groups of multiple sentence forms
        processTemporaryComponents(temporaryComponents,
                                   temporaryNegations,
                                   components,
                                   negations,
                                   trueComponent,
                                   falseComponent);
        addFormToCompletedValues(form, completedSentenceFormValues, components);
      }
    }

    LOGGER.trace("Final function info");
    for (FunctionInfo lInfo : functionInfoMap.values())
    {
      LOGGER.trace(lInfo.toString());
    }

    //Connect "next" to "true"
    LOGGER.trace("Adding transitions...");
    addTransitions(components, xiComponentFactory);
    //Set up "init" proposition
    LOGGER.trace("Setting up 'init' proposition...");
    setUpInit(components, trueComponent, falseComponent, xiComponentFactory);
    //Now we can safely...
    LOGGER.trace("Num components before useless removed: " + components.size());

    removeUselessBasePropositions(components,
                                  negations,
                                  trueComponent,
                                  falseComponent);
    LOGGER.trace("Num components after useless removed: " + components.size());
    LOGGER.trace("Creating component set...");
    Set<PolymorphicComponent> componentSet = new HashSet<>(components.values());
    //Try saving some memory here...
    components = null;
    negations = null;
    completeComponentSet(componentSet);
    ConcurrencyUtils.checkForInterruption();
    LOGGER.trace("Initializing propnet object...");
    //Make it look the same as the PropNetFactory results, until we decide
    //how we want it to look
    normalizePropositions(componentSet);
    PolymorphicPropNet propnet = xiComponentFactory.createPropNet(roles, componentSet);

    LOGGER.trace("Num components at end of propnet construction (pre-optimization): " + componentSet.size());

    return propnet;
  }


  private static void removeUselessBasePropositions(Map<GdlSentence, PolymorphicComponent> components,
                                                    Map<GdlSentence, PolymorphicComponent> negations,
                                                    PolymorphicConstant trueComponent,
                                                    PolymorphicConstant falseComponent)
      throws InterruptedException
  {
    boolean changedSomething = false;
    for (Entry<GdlSentence, PolymorphicComponent> entry : components.entrySet())
    {
      if (entry.getKey().getName() == GdlPool.TRUE)
      {
        PolymorphicComponent comp = entry.getValue();
        if (comp.getInputs().size() == 0)
        {
          comp.addInput(falseComponent);
          falseComponent.addOutput(comp);
          changedSomething = true;
        }
      }
    }
    if (!changedSomething)
      return;

    optimizeAwayTrueAndFalse(components,
                             negations,
                             trueComponent,
                             falseComponent);
  }

  /**
   * Changes the propositions contained in the propnet so that they correspond
   * to the outputs of the PropNetFactory. This is for consistency and for
   * backwards compatibility with respect to state machines designed for the
   * old propnet factory. Feel free to remove this for your player.
   *
   * @param componentSet
   */
  private static void normalizePropositions(Set<PolymorphicComponent> componentSet)
  {
    for (PolymorphicComponent component : componentSet)
    {
      if (component instanceof PolymorphicProposition)
      {
        PolymorphicProposition p = (PolymorphicProposition)component;
        GdlSentence sentence = p.getName();
        if (sentence instanceof GdlRelation)
        {
          GdlRelation relation = (GdlRelation)sentence;
          if (relation.getName().equals(GdlPool.NEXT))
          {
            p.setName(GdlPool.getProposition(GdlPool.getConstant("anon")));
          }
        }
      }
    }
  }

  private static void addFormToCompletedValues(SentenceForm form,
                                               Map<SentenceForm, Collection<GdlSentence>> completedSentenceFormValues,
                                               ConstantChecker constantChecker)
  {
    List<GdlSentence> sentences = new ArrayList<>();
    sentences.addAll(constantChecker.getTrueSentences(form));

    completedSentenceFormValues.put(form, sentences);
  }


  private static void addFormToCompletedValues(SentenceForm form,
                                               Map<SentenceForm, Collection<GdlSentence>> completedSentenceFormValues,
                                               Map<GdlSentence, PolymorphicComponent> components)
      throws InterruptedException
  {
    //Kind of inefficient. Could do better by collecting these as we go,
    //then adding them back into the CSFV map once the sentence forms are complete.
    //completedSentenceFormValues.put(form, new ArrayList<GdlSentence>());
    List<GdlSentence> sentences = new ArrayList<>();
    for (GdlSentence sentence : components.keySet())
    {
      ConcurrencyUtils.checkForInterruption();
      if (form.matches(sentence))
      {
        //The sentence has a node associated with it
        sentences.add(sentence);
      }
    }
    completedSentenceFormValues.put(form, sentences);
  }


  private static void addConstantsToFunctionInfo(SentenceForm form,
                                                 ConstantChecker constantChecker,
                                                 Map<SentenceForm, FunctionInfo> functionInfoMap)
      throws InterruptedException
  {
    functionInfoMap.put(form, FunctionInfoImpl.create(form, constantChecker));
  }


  private static void processTemporaryComponents(Map<GdlSentence, PolymorphicComponent> temporaryComponents,
                                                 Map<GdlSentence, PolymorphicComponent> temporaryNegations,
                                                 Map<GdlSentence, PolymorphicComponent> components,
                                                 Map<GdlSentence, PolymorphicComponent> negations,
                                                 PolymorphicComponent trueComponent,
                                                 PolymorphicComponent falseComponent)
      throws InterruptedException
  {
    //For each component in temporary components, we want to "put it back"
    //into the main components section.
    //We also want to do optimization here...
    //We don't want to end up with anything following from true/false.

    //Everything following from a temporary component (its outputs)
    //should instead become an output of the actual component.
    //If there is no actual component generated, then the statement
    //is necessarily FALSE and should be replaced by the false
    //component.
    for (GdlSentence sentence : temporaryComponents.keySet())
    {
      PolymorphicComponent tempComp = temporaryComponents.get(sentence);
      PolymorphicComponent realComp = components.get(sentence);
      if (realComp == null)
      {
        realComp = falseComponent;
      }
      for (PolymorphicComponent output : tempComp.getOutputs())
      {
        //Disconnect
        output.removeInput(tempComp);
        //tempComp.removeOutput(output); //do at end
        //Connect
        output.addInput(realComp);
        realComp.addOutput(output);
      }
      tempComp.removeAllOutputs();

      if (temporaryNegations.containsKey(sentence))
      {
        //Should be pointing to a "not" that now gets input from realComp
        //Should be fine to put into negations
        negations.put(sentence, temporaryNegations.get(sentence));
        //If this follows true/false, will get resolved by the next set of optimizations
      }

      optimizeAwayTrueAndFalse(components,
                               negations,
                               trueComponent,
                               falseComponent);

    }
  }

  /**
   * Components and negations may be null, if e.g. this is a post-optimization.
   * TrueComponent and falseComponent are required. Doesn't actually work that
   * way... shoot. Need something that will remove the component from the
   * propnet entirely.
   *
   * @throws InterruptedException
   */
  private static void optimizeAwayTrueAndFalse(Map<GdlSentence, PolymorphicComponent> components,
                                               Map<GdlSentence, PolymorphicComponent> negations,
                                               PolymorphicComponent trueComponent,
                                               PolymorphicComponent falseComponent)
      throws InterruptedException
  {
    while (hasNonessentialChildren(trueComponent, false) ||
           hasNonessentialChildren(falseComponent, true))
    {
      ConcurrencyUtils.checkForInterruption();
      optimizeAwayTrue(components,
                       negations,
                       null,
                       trueComponent,
                       falseComponent);
      optimizeAwayFalse(components,
                        negations,
                        null,
                        trueComponent,
                        falseComponent);
    }
  }

  private static void optimizeAwayTrueAndFalse(PolymorphicPropNet pn,
                                               PolymorphicComponent trueComponent,
                                               PolymorphicComponent falseComponent)
  {
    while (hasNonessentialChildren(trueComponent, false) ||
           hasNonessentialChildren(falseComponent, true))
    {
      optimizeAwayTrue(null, null, pn, trueComponent, falseComponent);
      optimizeAwayFalse(null, null, pn, trueComponent, falseComponent);
    }
  }

  private static Set<PolymorphicComponent> findImmediatelyNonEssentialChildren(PolymorphicComponent parent,
                                                                               boolean forFalse)
  {
    Set<PolymorphicComponent> result = new HashSet<>();

    for (PolymorphicComponent c : parent.getOutputs())
    {
      if (!isEssentialProposition(c, forFalse) &&
          !(c instanceof PolymorphicTransition))
      {
        result.add(c);
      }
    }
    return result;
  }

  //TODO: Create a version with just a set of components that we can share with post-optimizations
  private static void optimizeAwayFalse(Map<GdlSentence, PolymorphicComponent> components,
                                        Map<GdlSentence, PolymorphicComponent> negations,
                                        PolymorphicPropNet pn,
                                        PolymorphicComponent trueComponent,
                                        PolymorphicComponent falseComponent)
  {
    assert ((components != null && negations != null) || pn != null);
    assert ((components == null && negations == null) || pn == null);

    Set<PolymorphicComponent> nonEssentials;

    //while(hasNonessentialChildren(falseComponent)) {
    while ((nonEssentials = findImmediatelyNonEssentialChildren(falseComponent,
                                                                true)).size() != 0)
    {
      //int count = 0;
      for (PolymorphicComponent output : nonEssentials)
      {
        //Iterator<Component> outputItr = falseComponent.getOutputs().iterator();
        //Component output = outputItr.next();
        //while(isEssentialProposition(output) || output instanceof Transition) {
        //    if(outputItr.hasNext())
        //        output = outputItr.next();
        //    else
        //        return;
        //}
        if (output instanceof PolymorphicProposition)
        {
          //Move its outputs to be outputs of false
          for (PolymorphicComponent child : output.getOutputs())
          {
            //Disconnect
            child.removeInput(output);
            //output.removeOutput(child); //do at end
            //Reconnect; will get children before returning, if nonessential
            falseComponent.addOutput(child);
            child.addInput(falseComponent);
          }
          output.removeAllOutputs();

          if (!isEssentialProposition(output, true))
          {
            PolymorphicProposition prop = (PolymorphicProposition)output;
            //Remove the proposition entirely
            falseComponent.removeOutput(output);
            output.removeInput(falseComponent);
            //Update its location to the trueComponent in our map
            if (components != null)
            {
              components.put(prop.getName(), falseComponent);
              negations.put(prop.getName(), trueComponent);
            }
            else
            {
              pn.removeComponent(output);
            }
          }
        }
        else if (output instanceof PolymorphicAnd)
        {
          PolymorphicAnd and = (PolymorphicAnd)output;
          and.removeInput(falseComponent);
          falseComponent.removeOutput(and);
          //Attach children of and to falseComponent
          for (PolymorphicComponent child : and.getOutputs())
          {
            child.addInput(falseComponent);
            falseComponent.addOutput(child);
            child.removeInput(and);
          }
          //Disconnect and completely
          and.removeAllOutputs();
          for (PolymorphicComponent parent : and.getInputs())
            parent.removeOutput(and);
          and.removeAllInputs();
          if (pn != null)
            pn.removeComponent(and);
        }
        else if (output instanceof PolymorphicOr)
        {
          PolymorphicOr or = (PolymorphicOr)output;
          //Remove as input from or
          or.removeInput(falseComponent);
          falseComponent.removeOutput(or);
          //If or has only one input, remove it
          if (or.getInputs().size() == 1)
          {
            PolymorphicComponent in = or.getSingleInput();
            or.removeInput(in);
            in.removeOutput(or);
            for (PolymorphicComponent out : or.getOutputs())
            {
              //Disconnect from and
              out.removeInput(or);
              //or.removeOutput(out); //do at end
              //Connect directly to the new input
              out.addInput(in);
              in.addOutput(out);
            }
            or.removeAllOutputs();
            if (pn != null)
              pn.removeComponent(or);
          }
        }
        else if (output instanceof PolymorphicNot)
        {
          PolymorphicNot not = (PolymorphicNot)output;
          //Disconnect from falseComponent
          not.removeInput(falseComponent);
          falseComponent.removeOutput(not);
          //Connect all children of the not to trueComponent
          for (PolymorphicComponent child : not.getOutputs())
          {
            //Disconnect
            child.removeInput(not);
            //not.removeOutput(child); //Do at end
            //Connect to trueComponent
            child.addInput(trueComponent);
            trueComponent.addOutput(child);
          }
          not.removeAllOutputs();
          if (pn != null)
            pn.removeComponent(not);
        }
        else if (output instanceof PolymorphicTransition)
        {
          //???
          System.err.println("Fix optimizeAwayFalse's case for Transitions");
        }
      }
    }
  }


  private static void optimizeAwayTrue(Map<GdlSentence, PolymorphicComponent> components,
                                       Map<GdlSentence, PolymorphicComponent> negations,
                                       PolymorphicPropNet pn,
                                       PolymorphicComponent trueComponent,
                                       PolymorphicComponent falseComponent)
  {
    assert ((components != null && negations != null) || pn != null);

    Set<PolymorphicComponent> nonEssentials;

    //while(hasNonessentialChildren(trueComponent)) {
    while ((nonEssentials = findImmediatelyNonEssentialChildren(trueComponent,
                                                                false)).size() != 0)
    {
      //int count = 0;
      for (PolymorphicComponent output : nonEssentials)
      {
        //Iterator<Component> outputItr = trueComponent.getOutputs().iterator();
        //Component output = outputItr.next();
        //while(isEssentialProposition(output) || output instanceof Transition) {
        //	if (outputItr.hasNext())
        //		output = outputItr.next();
        //	else
        //		return;
        //}
        if (output instanceof PolymorphicProposition)
        {
          //Move its outputs to be outputs of true
          for (PolymorphicComponent child : output.getOutputs())
          {
            //Disconnect
            child.removeInput(output);
            //output.removeOutput(child); //do at end
            //Reconnect; will get children before returning, if nonessential
            trueComponent.addOutput(child);
            child.addInput(trueComponent);
          }
          output.removeAllOutputs();

          if (!isEssentialProposition(output, false))
          {
            PolymorphicProposition prop = (PolymorphicProposition)output;
            //Remove the proposition entirely
            trueComponent.removeOutput(output);
            output.removeInput(trueComponent);
            //Update its location to the trueComponent in our map
            if (components != null)
            {
              components.put(prop.getName(), trueComponent);
              negations.put(prop.getName(), falseComponent);
            }
            else
            {
              pn.removeComponent(output);
            }
          }
        }
        else if (output instanceof PolymorphicOr)
        {
          PolymorphicOr or = (PolymorphicOr)output;
          or.removeInput(trueComponent);
          trueComponent.removeOutput(or);
          //Attach children of or to trueComponent
          for (PolymorphicComponent child : or.getOutputs())
          {
            child.addInput(trueComponent);
            trueComponent.addOutput(child);
            child.removeInput(or);
          }
          //Disconnect or completely
          or.removeAllOutputs();
          for (PolymorphicComponent parent : or.getInputs())
            parent.removeOutput(or);
          or.removeAllInputs();
          if (pn != null)
            pn.removeComponent(or);
        }
        else if (output instanceof PolymorphicAnd)
        {
          PolymorphicAnd and = (PolymorphicAnd)output;
          //Remove as input from and
          and.removeInput(trueComponent);
          trueComponent.removeOutput(and);
          //If and has only one input, remove it
          if (and.getInputs().size() == 1)
          {
            PolymorphicComponent in = and.getSingleInput();
            and.removeInput(in);
            in.removeOutput(and);
            for (PolymorphicComponent out : and.getOutputs())
            {
              //Disconnect from and
              out.removeInput(and);
              //and.removeOutput(out); //do at end
              //Connect directly to the new input
              out.addInput(in);
              in.addOutput(out);
            }
            and.removeAllOutputs();
            if (pn != null)
              pn.removeComponent(and);
          }
        }
        else if (output instanceof PolymorphicNot)
        {
          PolymorphicNot not = (PolymorphicNot)output;
          //Disconnect from trueComponent
          not.removeInput(trueComponent);
          trueComponent.removeOutput(not);
          //Connect all children of the not to falseComponent
          for (PolymorphicComponent child : not.getOutputs())
          {
            //Disconnect
            child.removeInput(not);
            //not.removeOutput(child); //Do at end
            //Connect to falseComponent
            child.addInput(falseComponent);
            falseComponent.addOutput(child);
          }
          not.removeAllOutputs();
          if (pn != null)
            pn.removeComponent(not);
        }
        else if (output instanceof PolymorphicTransition)
        {
          //???
          System.err.println("Fix optimizeAwayTrue's case for Transitions");
        }
      }
    }
  }


  private static boolean hasNonessentialChildren(PolymorphicComponent trueComponent,
                                                 boolean forFalse)
  {
    for (PolymorphicComponent child : trueComponent.getOutputs())
    {
      if (child instanceof PolymorphicTransition)
        continue;
      if (!isEssentialProposition(child, forFalse))
        return true;
      //We don't want any grandchildren, either
      //	SD - this breaks down in certain case leading to infinite loops - for
      //	example when an isLegal is used as input to other parts of the state
      //	calculation (crossers3 is an example such ruleset)
      //if(!child.getOutputs().isEmpty())
      //	return true;
    }
    return false;
  }


  private static boolean isEssentialProposition(PolymorphicComponent component,
                                                boolean forFalse)
  {
    if (!(component instanceof PolymorphicProposition))
      return false;

    //We're looking for things that would be outputs of "true" or "false",
    //but we would still want to keep as propositions to be read by the
    //state machine
    PolymorphicProposition prop = (PolymorphicProposition)component;
    GdlConstant name = prop.getName().getName();

    return (name.equals(GdlPool.LEGAL) && !forFalse) /* || name.equals(NEXT) */||
           name.equals(GdlPool.GOAL) || name.equals(GdlPool.INIT) || name.equals(GdlPool.TERMINAL);
  }


  private static void completeComponentSet(Set<PolymorphicComponent> componentSet)
  {
    Set<PolymorphicComponent> newComponents = new HashSet<>();
    Set<PolymorphicComponent> componentsToTry = new HashSet<>(componentSet);
    while (!componentsToTry.isEmpty())
    {
      for (PolymorphicComponent c : componentsToTry)
      {
        for (PolymorphicComponent out : c.getOutputs())
        {
          if (!componentSet.contains(out))
            newComponents.add(out);
        }
        for (PolymorphicComponent in : c.getInputs())
        {
          if (!componentSet.contains(in))
            newComponents.add(in);
        }
      }
      componentSet.addAll(newComponents);
      componentsToTry = newComponents;
      newComponents = new HashSet<>();
    }
  }


  private static void addTransitions(Map<GdlSentence, PolymorphicComponent> components,
                                     PolymorphicComponentFactory componentFactory)
  {
    for (Entry<GdlSentence, PolymorphicComponent> entry : components
        .entrySet())
    {
      GdlSentence sentence = entry.getKey();

      if (sentence.getName().equals(GdlPool.NEXT))
      {
        //connect to true
        GdlSentence trueSentence = GdlPool.getRelation(GdlPool.TRUE, sentence.getBody());
        PolymorphicComponent nextComponent = entry.getValue();
        PolymorphicComponent trueComponent = components.get(trueSentence);
        //There might be no true component (for example, because the bases
        //told us so). If that's the case, don't have a transition.
        if (trueComponent == null)
        {
          // Skipping transition to supposedly impossible 'trueSentence'
          continue;
        }
        PolymorphicTransition transition = componentFactory
            .createTransition(-1);
        transition.addInput(nextComponent);
        nextComponent.addOutput(transition);
        transition.addOutput(trueComponent);
        trueComponent.addInput(transition);
      }
    }
  }

  //TODO: Replace with version using constantChecker only
  //TODO: This can give problematic results if interpreted in
  //the standard way (see test_case_3d)
  private static void setUpInit(Map<GdlSentence, PolymorphicComponent> components,
                                PolymorphicConstant trueComponent,
                                PolymorphicConstant falseComponent,
                                PolymorphicComponentFactory componentFactory)
  {
    PolymorphicProposition initProposition = componentFactory.createProposition(-1, GdlPool.getProposition(INIT_CAPS));
    for (Entry<GdlSentence, PolymorphicComponent> entry : components.entrySet())
    {
      //Is this something that will be true?
      if (entry.getValue() == trueComponent)
      {
        if (entry.getKey().getName().equals(GdlPool.INIT))
        {
          //Find the corresponding true sentence
          GdlSentence trueSentence = GdlPool.getRelation(GdlPool.TRUE, entry.getKey().getBody());
          PolymorphicComponent trueSentenceComponent = components.get(trueSentence);
          if (trueSentenceComponent.getInputs().isEmpty())
          {
            //Case where there is no transition input
            //Add the transition input, connect to init, continue loop
            PolymorphicTransition transition = componentFactory.createTransition(-1);
            //init goes into transition
            transition.addInput(initProposition);
            initProposition.addOutput(transition);
            //transition goes into component
            trueSentenceComponent.addInput(transition);
            transition.addOutput(trueSentenceComponent);
          }
          else
          {
            //The transition already exists
            PolymorphicComponent transition = trueSentenceComponent.getSingleInput();

            //We want to add init as a thing that precedes the transition
            //Disconnect existing input
            PolymorphicComponent input = transition.getSingleInput();
            //input and init go into or, or goes into transition
            input.removeOutput(transition);
            transition.removeInput(input);
            List<PolymorphicComponent> orInputs = new ArrayList<>(2);
            orInputs.add(input);
            orInputs.add(initProposition);
            orify(orInputs, transition, falseComponent, componentFactory);
          }
        }
      }
    }
  }

  /**
   * Adds an or gate connecting the inputs to produce the output. Handles
   * special optimization cases like a true/false input.
   */
  private static void orify(Collection<PolymorphicComponent> inputs,
                            PolymorphicComponent output,
                            PolymorphicConstant falseProp,
                            PolymorphicComponentFactory componentFactory)
  {
    //TODO: Look for already-existing ors with the same inputs?
    //Or can this be handled with a GDL transformation?

    //Special case: An input is the true constant
    for (PolymorphicComponent in : inputs)
    {
      if (in instanceof PolymorphicConstant && in.getValue())
      {
        //True constant: connect that to the component, done
        in.addOutput(output);
        output.addInput(in);
        return;
      }
    }

    //Special case: An input is "or"
    //I'm honestly not sure how to handle special cases here...
    //What if that "or" gate has multiple outputs? Could that happen?

    //For reals... just skip over any false constants
    PolymorphicOr or = componentFactory.createOr(-1, -1);
    for (PolymorphicComponent in : inputs)
    {
      if (!(in instanceof PolymorphicConstant))
      {
        in.addOutput(or);
        or.addInput(in);
      }
    }
    //What if they're all false? (Or inputs is empty?) Then no inputs at this point...
    if (or.getInputs().isEmpty())
    {
      //Hook up to "false"
      falseProp.addOutput(output);
      output.addInput(falseProp);
      return;
    }
    //If there's just one, on the other hand, don't use the or gate
    if (or.getInputs().size() == 1)
    {
      PolymorphicComponent in = or.getSingleInput();
      in.removeOutput(or);
      or.removeInput(in);
      in.addOutput(output);
      output.addInput(in);
      return;
    }
    or.addOutput(output);
    output.addInput(or);
  }

  //TODO: This code is currently used by multiple classes, so perhaps it should be
  //factored out into the SentenceModel.
  private static List<SentenceForm> getTopologicalOrdering(Set<SentenceForm> forms,
                                                           Multimap<SentenceForm, SentenceForm> dependencyGraph,
                                                           boolean usingBase,
                                                           boolean usingInput)
      throws InterruptedException
  {
    //We want each form as a key of the dependency graph to
    //follow all the forms in the dependency graph, except maybe itself
    Queue<SentenceForm> queue = new LinkedList<>(forms);
    List<SentenceForm> ordering = new ArrayList<>(forms.size());
    Set<SentenceForm> alreadyOrdered = new HashSet<>();

    int processingSequence = 0;
    int lastProcessedSequence = 0;

    while (!queue.isEmpty())
    {
      boolean looping = (processingSequence - lastProcessedSequence) > queue.size();
      SentenceForm curForm = queue.remove();
      boolean readyToAdd = true;
      //Don't add if there are dependencies
      for (SentenceForm dependency : dependencyGraph.get(curForm))
      {
        if (!dependency.equals(curForm) &&
            !alreadyOrdered.contains(dependency))
        {
          readyToAdd = false;
          break;
        }
      }
      //Don't add if it's true/next/legal/does and we're waiting for base/input
      if (usingBase &&
          (curForm.getName().equals(GdlPool.TRUE) || curForm.getName().equals(GdlPool.NEXT) || curForm
              .getName().equals(GdlPool.INIT)))
      {
        //Have we added the corresponding base sf yet?
        SentenceForm baseForm = curForm.withName(GdlPool.BASE);
        if (!alreadyOrdered.contains(baseForm))
        {
          //  If we're looping here it's probably a GDL issue - flag up where we're failing
          //  and try to process assuming the proposition will exist
          if ( looping )
          {
            LOGGER.warn("Missing base form.  Unable to process: " + curForm);
          }
          else
          {
            readyToAdd = false;
          }
        }
      }
      if (usingInput &&
          (curForm.getName().equals(GdlPool.DOES) || curForm.getName().equals(GdlPool.LEGAL)))
      {
        SentenceForm inputForm = curForm.withName(GdlPool.INPUT);
        if (!alreadyOrdered.contains(inputForm))
        {
          if ( looping )
          {
            LOGGER.warn("Missing input form.  Unable to process: " + curForm);
          }
          else
          {
            readyToAdd = false;
          }
        }
      }
      //Add it
      processingSequence++;
      if (readyToAdd)
      {
        lastProcessedSequence = processingSequence;
        ordering.add(curForm);
        alreadyOrdered.add(curForm);
      }
      else
      {
        queue.add(curForm);
      }
      //TODO: Add check for an infinite loop here, or stratify loops

      ConcurrencyUtils.checkForInterruption();
    }
    return ordering;
  }

  private static void addSentenceForm(SentenceForm form,
                                      SentenceDomainModel model,
                                      Map<GdlSentence, PolymorphicComponent> components,
                                      Map<GdlSentence, PolymorphicComponent> negations,
                                      PolymorphicConstant trueComponent,
                                      PolymorphicConstant falseComponent,
                                      boolean usingBase,
                                      boolean usingInput,
                                      Set<SentenceForm> recursionForms,
                                      Map<GdlSentence, PolymorphicComponent> temporaryComponents,
                                      Map<GdlSentence, PolymorphicComponent> temporaryNegations,
                                      Map<SentenceForm, FunctionInfo> functionInfoMap,
                                      ConstantChecker constantChecker,
                                      Map<SentenceForm, Collection<GdlSentence>> completedSentenceFormValues,
                                      PolymorphicComponentFactory componentFactory)
      throws InterruptedException
  {
    //This is the meat of it (along with the entire Assignments class).
    //We need to enumerate the possible propositions in the sentence form...
    //We also need to hook up the sentence form to the inputs that can make it true.
    //We also try to optimize as we go, which means possibly removing the
    //proposition if it isn't actually possible, or replacing it with
    //true/false if it's a constant.
    Set<GdlSentence> alwaysTrueSentences = model
        .getSentencesListedAsTrue(form);
    Set<GdlRule> rules = model.getRules(form);

    for (GdlSentence alwaysTrueSentence : alwaysTrueSentences)
    {
      //We add the sentence as a constant
      if (alwaysTrueSentence.getName().equals(GdlPool.LEGAL) ||
          alwaysTrueSentence.getName().equals(GdlPool.NEXT) ||
          alwaysTrueSentence.getName().equals(GdlPool.GOAL))
      {
        PolymorphicProposition prop = componentFactory.createProposition(-1, alwaysTrueSentence);
        //Attach to true
        trueComponent.addOutput(prop);
        prop.addInput(trueComponent);
        //Still want the same components;
        //we just don't want this to be anonymized
      }
      //Assign as true
      components.put(alwaysTrueSentence, trueComponent);
      negations.put(alwaysTrueSentence, falseComponent);
      continue;
    }

    //For does/true, make nodes based on input/base, if available
    if (usingInput && form.getName().equals(GdlPool.DOES))
    {
      //Add only those propositions for which there is a corresponding INPUT
      SentenceForm inputForm = form.withName(GdlPool.INPUT);
      for (GdlSentence inputSentence : constantChecker
          .getTrueSentences(inputForm))
      {
        GdlSentence doesSentence = GdlPool.getRelation(GdlPool.DOES, inputSentence.getBody());
        PolymorphicProposition prop = componentFactory.createProposition(-1, doesSentence);
        components.put(doesSentence, prop);
      }
      return;
    }
    if (usingBase && form.getName().equals(GdlPool.TRUE))
    {
      SentenceForm baseForm = form.withName(GdlPool.BASE);
      for (GdlSentence baseSentence : constantChecker
          .getTrueSentences(baseForm))
      {
        GdlSentence trueSentence = GdlPool.getRelation(GdlPool.TRUE, baseSentence.getBody());
        PolymorphicProposition prop = componentFactory
            .createProposition(-1, trueSentence);
        components.put(trueSentence, prop);
      }
      return;
    }

    Map<GdlSentence, Set<PolymorphicComponent>> inputsToOr = new HashMap<>();
    for (GdlRule rule : rules)
    {
      Assignments assignments = AssignmentsFactory
          .getAssignmentsForRule(rule,
                                 model,
                                 functionInfoMap,
                                 completedSentenceFormValues);

      //Calculate vars in live (non-constant, non-distinct) conjuncts
      Set<GdlVariable> varsInLiveConjuncts = getVarsInLiveConjuncts(rule,
                                                                    constantChecker
                                                                        .getConstantSentenceForms());
      varsInLiveConjuncts.addAll(GdlUtils.getVariables(rule.getHead()));
      Set<GdlVariable> varsInRule = new HashSet<>(GdlUtils.getVariables(rule));
      boolean preventDuplicatesFromConstants = (varsInRule.size() > varsInLiveConjuncts
          .size());

      //Do we just pass those to the Assignments class in that case?
      for (AssignmentIterator asnItr = assignments.getIterator(); asnItr
          .hasNext();)
      {
        Map<GdlVariable, GdlConstant> assignment = asnItr.next();
        if (assignment == null)
          continue; //Not sure if this will ever happen

        ConcurrencyUtils.checkForInterruption();

        GdlSentence sentence = CommonTransforms.replaceVariables(rule
            .getHead(), assignment);

        //Now we go through the conjuncts as before, but we wait to hook them up.
        List<PolymorphicComponent> componentsToConnect = new ArrayList<>(rule
            .arity());
        for (GdlLiteral literal : rule.getBody())
        {
          if (literal instanceof GdlSentence)
          {
            //Get the sentence post-substitutions
            GdlSentence transformed = CommonTransforms
                .replaceVariables((GdlSentence)literal, assignment);

            //Check for constant-ness
            SentenceForm conjunctForm = model.getSentenceForm(transformed);
            if (constantChecker.isConstantForm(conjunctForm))
            {
              if (!constantChecker.isTrueConstant(transformed))
              {
                List<GdlVariable> varsToChange = getVarsInConjunct(literal);
                asnItr.changeOneInNext(varsToChange, assignment);
                componentsToConnect.add(null);
              }
              continue;
            }

            PolymorphicComponent conj = components.get(transformed);
            //If conj is null and this is a sentence form we're still handling,
            //hook up to a temporary sentence form
            if (conj == null)
            {
              conj = temporaryComponents.get(transformed);
            }
            if (conj == null &&
                SentenceModelUtils.inSentenceFormGroup(transformed,
                                                       recursionForms))
            {
              //Set up a temporary component
              PolymorphicProposition tempProp = componentFactory
                  .createProposition(-1, transformed);
              temporaryComponents.put(transformed, tempProp);
              conj = tempProp;
            }
            //Let's say this is false; we want to backtrack and change the right variable
            if (conj == null || isThisConstant(conj, falseComponent))
            {
              List<GdlVariable> varsInConjunct = getVarsInConjunct(literal);
              asnItr.changeOneInNext(varsInConjunct, assignment);
              //These last steps just speed up the process
              //telling the factory to ignore this rule
              componentsToConnect.add(null);
              continue; //look at all the other restrictions we'll face
            }

            componentsToConnect.add(conj);
          }
          else if (literal instanceof GdlNot)
          {
            //Add a "not" if necessary
            //Look up the negation
            GdlSentence internal = (GdlSentence)((GdlNot)literal).getBody();
            GdlSentence transformed = CommonTransforms
                .replaceVariables(internal, assignment);

            //Add constant-checking here...
            SentenceForm conjunctForm = model.getSentenceForm(transformed);
            if (constantChecker.isConstantForm(conjunctForm))
            {
              if (constantChecker.isTrueConstant(transformed))
              {
                List<GdlVariable> varsToChange = getVarsInConjunct(literal);
                asnItr.changeOneInNext(varsToChange, assignment);
                componentsToConnect.add(null);
              }
              continue;
            }

            PolymorphicComponent conj = negations.get(transformed);
            if (isThisConstant(conj, falseComponent))
            {
              //We need to change one of the variables inside
              List<GdlVariable> varsInConjunct = getVarsInConjunct(internal);
              asnItr.changeOneInNext(varsInConjunct, assignment);
              //ignore this rule
              componentsToConnect.add(null);
              continue;
            }
            if (conj == null)
            {
              conj = temporaryNegations.get(transformed);
            }
            //Check for the recursive case:
            if (conj == null &&
                SentenceModelUtils.inSentenceFormGroup(transformed,
                                                       recursionForms))
            {
              PolymorphicComponent positive = components.get(transformed);
              if (positive == null)
              {
                positive = temporaryComponents.get(transformed);
              }
              if (positive == null)
              {
                //Make the temporary proposition
                PolymorphicProposition tempProp = componentFactory
                    .createProposition(-1, transformed);
                temporaryComponents.put(transformed, tempProp);
                positive = tempProp;
              }
              //Positive is now set and in temporaryComponents
              //Evidently, wasn't in temporaryNegations
              //So we add the "not" gate and set it in temporaryNegations
              PolymorphicNot not = componentFactory.createNot(-1);
              //Add positive as input
              not.addInput(positive);
              positive.addOutput(not);
              temporaryNegations.put(transformed, not);
              conj = not;
            }
            if (conj == null)
            {
              PolymorphicComponent positive = components.get(transformed);
              //No, because then that will be attached to "negations", which could be bad

//              if ( positive == null && transformed.arity() == 0 )
//              {
//                 positive = components.get(GdlPool.getProposition(transformed.getName()));
//              }
              if (positive == null)
              {
                //So the positive can't possibly be true (unless we have recursion)
                //and so this would be positive always
                //We want to just skip this conjunct, so we continue to the next

                continue; //to the next conjunct
              }

              //Check if we're sharing a component with another sentence with a negation
              //(i.e. look for "nots" in our outputs and use those instead)
              PolymorphicNot existingNotOutput = getNotOutput(positive);
              if (existingNotOutput != null)
              {
                componentsToConnect.add(existingNotOutput);
                negations.put(transformed, existingNotOutput);
                continue; //to the next conjunct
              }

              PolymorphicNot not = componentFactory.createNot(-1);
              not.addInput(positive);
              positive.addOutput(not);
              negations.put(transformed, not);
              conj = not;
            }
            componentsToConnect.add(conj);
          }
          else if (literal instanceof GdlDistinct)
          {
            //Already handled; ignore
          }
          else
          {
            throw new RuntimeException("Unwanted GdlLiteral type");
          }
        }
        if (!componentsToConnect.contains(null))
        {
          //Connect all the components
          PolymorphicProposition andComponent = componentFactory
              .createProposition(-1, TEMP);

          andify(componentsToConnect,
                 andComponent,
                 trueComponent,
                 componentFactory);
          if (!isThisConstant(andComponent, falseComponent))
          {
            if (!inputsToOr.containsKey(sentence))
              inputsToOr.put(sentence, new HashSet<PolymorphicComponent>());
            inputsToOr.get(sentence).add(andComponent);
            //We'll want to make sure at least one of the non-constant
            //components is changing
            if (preventDuplicatesFromConstants)
            {
              asnItr.changeOneInNext(varsInLiveConjuncts, assignment);
            }
          }
        }
      }
    }

    //At the end, we hook up the conjuncts
    for (Entry<GdlSentence, Set<PolymorphicComponent>> entry : inputsToOr
        .entrySet())
    {
      ConcurrencyUtils.checkForInterruption();

      GdlSentence sentence = entry.getKey();
      Set<PolymorphicComponent> inputs = entry.getValue();
      Set<PolymorphicComponent> realInputs = new HashSet<>();
      for (PolymorphicComponent input : inputs)
      {
        if (input instanceof PolymorphicConstant ||
            input.getInputs().size() == 0)
        {
          realInputs.add(input);
        }
        else
        {
          realInputs.add(input.getSingleInput());
          input.getSingleInput().removeOutput(input);
          input.removeAllInputs();
        }
      }

      PolymorphicProposition prop = componentFactory
          .createProposition(-1, sentence);
      orify(realInputs, prop, falseComponent, componentFactory);
      components.put(sentence, prop);
    }

    //True/does sentences will have none of these rules, but
    //still need to exist/"float"
    //We'll do this if we haven't used base/input as a basis
    if (form.getName().equals(GdlPool.TRUE) || form.getName().equals(GdlPool.DOES))
    {
      for (GdlSentence sentence : model.getDomain(form))
      {
        ConcurrencyUtils.checkForInterruption();

        PolymorphicProposition prop = componentFactory
            .createProposition(-1, sentence);
        components.put(sentence, prop);
      }
    }

  }


  private static Set<GdlVariable> getVarsInLiveConjuncts(GdlRule rule,
                                                         Set<SentenceForm> constantSentenceForms)
  {
    Set<GdlVariable> result = new HashSet<>();
    for (GdlLiteral literal : rule.getBody())
    {
      if (literal instanceof GdlRelation)
      {
        if (!SentenceModelUtils.inSentenceFormGroup((GdlRelation)literal,
                                                    constantSentenceForms))
          result.addAll(GdlUtils.getVariables(literal));
      }
      else if (literal instanceof GdlNot)
      {
        GdlNot not = (GdlNot)literal;
        GdlSentence inner = (GdlSentence)not.getBody();
        if (!SentenceModelUtils.inSentenceFormGroup(inner,
                                                    constantSentenceForms))
          result.addAll(GdlUtils.getVariables(literal));
      }
    }
    return result;
  }

  private static boolean isThisConstant(PolymorphicComponent conj,
                                        PolymorphicConstant constantComponent)
  {
    if (conj == constantComponent)
      return true;
    return (conj instanceof PolymorphicProposition &&
            conj.getInputs().size() == 1 && conj.getSingleInput() == constantComponent);
  }


  private static PolymorphicNot getNotOutput(PolymorphicComponent positive)
  {
    for (PolymorphicComponent c : positive.getOutputs())
    {
      if (c instanceof PolymorphicNot)
      {
        return (PolymorphicNot)c;
      }
    }
    return null;
  }


  private static List<GdlVariable> getVarsInConjunct(GdlLiteral literal)
  {
    return GdlUtils.getVariables(literal);
  }


  private static void andify(List<PolymorphicComponent> inputs,
                             PolymorphicComponent output,
                             PolymorphicConstant trueProp,
                             PolymorphicComponentFactory componentFactory)
  {
    //Special case: If the inputs include false, connect false to thisComponent
    for (PolymorphicComponent c : inputs)
    {
      if (c instanceof PolymorphicConstant && !c.getValue())
      {
        //Connect false (c) to the output
        output.addInput(c);
        c.addOutput(output);
        return;
      }
    }

    //For reals... just skip over any true constants
    PolymorphicAnd and = componentFactory.createAnd(-1, -1);
    for (PolymorphicComponent in : inputs)
    {
      if (!(in instanceof PolymorphicConstant))
      {
        in.addOutput(and);
        and.addInput(in);
      }
    }
    //What if they're all true? (Or inputs is empty?) Then no inputs at this point...
    if (and.getInputs().isEmpty())
    {
      //Hook up to "true"
      trueProp.addOutput(output);
      output.addInput(trueProp);
      return;
    }
    //If there's just one, on the other hand, don't use the and gate
    if (and.getInputs().size() == 1)
    {
      PolymorphicComponent in = and.getSingleInput();
      in.removeOutput(and);
      and.removeInput(in);
      in.addOutput(output);
      output.addInput(in);
      return;
    }
    and.addOutput(output);
    output.addInput(and);
  }

  /**
   * Currently requires the init propositions to be left in the graph.
   *
   * @param pn
   */
  static enum Type {
    STAR(false, false, "grey"), TRUE(true, false, "green"), FALSE(false, true,
        "red"), BOTH(true, true, "white");
    private final boolean hasTrue;
    private final boolean hasFalse;
    private final String  color;

    Type(boolean hasTrue, boolean hasFalse, String color)
    {
      this.hasTrue = hasTrue;
      this.hasFalse = hasFalse;
      this.color = color;
    }

    public boolean hasTrue()
    {
      return hasTrue;
    }

    public boolean hasFalse()
    {
      return hasFalse;
    }

    public String getColor()
    {
      return color;
    }
  }

  static class ReEvaulationSet implements Iterable<PolymorphicComponent>
  {
    private Set<PolymorphicComponent>       contents = new LinkedHashSet<>();
    private Map<PolymorphicComponent, Type> reachability;

    public ReEvaulationSet(Map<PolymorphicComponent, Type> reachability)
    {
      this.reachability = reachability;
    }

    public void add(PolymorphicComponent c)
    {
      if (c instanceof PolymorphicConstant)
      {
        LOGGER.warn("Constant added to re-eval list!");
      }
      if (!reachability.containsKey(c))
      {
        LOGGER.warn("Unreachable component to re-eval list!");
      }

      contents.add(c);
    }

    public void addAll(Collection<? extends PolymorphicComponent> collection)
    {
      for (PolymorphicComponent c : collection)
      {
        add(c);
      }
    }

    public void remove(PolymorphicComponent c)
    {
      contents.remove(c);
    }

    @Override
    public Iterator<PolymorphicComponent> iterator()
    {
      return contents.iterator();
    }

    public boolean isEmpty()
    {
      return contents.isEmpty();
    }

    public int size()
    {
      return contents.size();
    }
  }

  private static void validateNoLeftBase(PolymorphicPropNet pn, Set<PolymorphicComponent> reachable)
  {
    for(PolymorphicProposition c: pn.getBasePropositions().values())
    {
      if ( reachable.contains(c) )
      {
        if (c.toString().contains("left"))
        {
          assert(false);
        }
        else
        {
          LOGGER.info("Include " + c);
        }
      }
    }
  }
  private static Set<PolymorphicComponent> findGoalReachableComponents(PolymorphicPropNet pn,
                                                                       PolymorphicProposition[] lOurGoals)
  {
    Set<PolymorphicComponent> directlyReachable = new HashSet<>();
    Set<PolymorphicComponent> result = new HashSet<>();

    recursiveFindReachable(pn, pn.getTerminalProposition(), directlyReachable, null, null, false);

    if ( lOurGoals != null )
    {
      Set<PolymorphicProposition> goalProxies = new HashSet<>();
      for( PolymorphicComponent c : lOurGoals)
      {
        //  For goals we don't want to consider propositions that are just proxies for the goals to
        //  be themselves directly reachable for the purposes of determining whether or not they are
        //  needed.  This allows self-support logic for base props that directly feed a specific goal
        //  to mask out dependencies on up-stream logic that is not otherwise required.  The justification
        //  for this is that at least one goal will always be true for each role (in a terminal state anyway)
        //  so there must be logic that is not simply self-support logic feeding goal proxies, through which
        //  dependencies will be traced
        while(c.getInputs().size()==1)
        {
          c = c.getSingleInput();
          if ( pn.getBasePropositions().values().contains(c) )
          {
            //  Stop at the first base prop we hit - this is a direct proxy for the goal
            break;
          }
        }

        recursiveFindReachable(pn, c, directlyReachable, null, null, false);

        //  If this was a base-prop proxy for a goal prop don't include it itself in the calculated direct
        //  dependencies (it will be included in the full [including indirect] dependencies calculated next,
        //  but not including it in this set means that self-support logic on it will not be traced back
        //  through)
        if ( pn.getBasePropositions().values().contains(c) )
        {
          goalProxies.add((PolymorphicProposition)c);
        }
      }
      directlyReachable.removeAll(goalProxies);
    }

    //  Now walk the propnet again, this time traversing move boundaries and trimming out
    //  self-dependencies for anything not already known to be directly reachable
    recursiveFindReachable(pn, pn.getTerminalProposition(), result, directlyReachable, null, true);

    if ( lOurGoals != null )
    {
      for( PolymorphicProposition c : lOurGoals)
      {
        recursiveFindReachable(pn, c, result, directlyReachable, null, true);
      }
    }

    return result;
  }

  /**
   * Remove bases that are not relevant to our play.  May also reduce
   * opponent roles to effective null roles (single psuedo-noop each turn) in
   * games which our result depends only on our play unconditionally on any choices
   * such an opponent might make
   * @param pn - propnet
   * @param ourRole - who we are (or null if no analysis relative to our role is required)
   * @param fillerMoveSet  - set to which all input propositions for moves which are only included
   *                         as virtual noops (have no direct impact on any base props related to our goals)
   *                         will have their GDL sentences added
   * @return true if the game can be treated as a puzzle (has no dependence on any other role's moves)
   */
  public static boolean removeIrrelevantBasesAndInputs(PolymorphicPropNet pn,
                                                       Role ourRole,
                                                       Set<GdlSentence> fillerMoveSet)
  {
    boolean isPseudoPuzzle = false;

    //  Any bases that cannot be reached by tracing back from either the terminal
    //  or a goal proposition are irrelevant (transiting through a virtual does->legal back-link)
    //  Note - this may leave some does props essentially unconnected, so this routine indirectly
    //  removes them also via subsequent trimming of then unconnected components, but this is
    //  not necessary explicitly here (and attempting to do so explicitly is actually somewhat
    //  tricky due to fairly common constructs involving distincts of move props)

    //  Initially just mark the components on which OUR goals depend if we have been given our role
    //  Don't try this if we only have a single goal value anyway - this is some stupid test game not
    //  a real game!
    PolymorphicProposition[] lOurGoals = null;
    if (ourRole != null)
    {
      lOurGoals = pn.getGoalPropositions().get(ourRole);
      if (lOurGoals == null)
      {
        LOGGER.error("Invalid GDL.  No goals defined for our role (" + ourRole + ")!  We will crash.");
      }
    }
    Set<PolymorphicComponent> reachableComponents = findGoalReachableComponents(pn, lOurGoals);

    if ((ourRole != null) && (lOurGoals.length > 1))
    {
      Set<PolymorphicComponent> coreReachableComponents = new HashSet<>();

      //  The core reachable components are those relevant directly to our goals
      coreReachableComponents.addAll(reachableComponents);

      if ( pn.getRoles().length > 1 )
      {
        //  Find the propnet's TRUE constant
        PolymorphicComponent trueConstant = null;

        for(PolymorphicComponent c : pn.getComponents())
        {
          if ( c instanceof PolymorphicConstant )
          {
            if ( ((PolymorphicConstant)c).getValue() )
            {
              trueConstant = c;
              break;
            }
          }
        }
        assert(trueConstant != null);

        //  For puzzles any moves that do not impact state on which the goals or terminality
        //  depend are irrelevant.  However, for non-puzzles we need to preserve them as they
        //  provide a possible source of pseudo-noops that could potentially result in
        //  Zugzwang in a multi-player game.  Hence explicitly preserve legals in non-puzzles
        //  for at least our role
        for( PolymorphicProposition c : pn.getLegalPropositions().get(ourRole))
        {
          recursiveFindReachable(pn, c, reachableComponents, reachableComponents, null, true);

          //  If a legal is not within the core reachable set it has no impact on anything
          //  that influences our goals and is therefore only relevant as a possible source of
          //  what amounts to a noop (in the subset of relevant components)
          if ( !coreReachableComponents.contains(c) )
          {
            PolymorphicProposition input = pn.getLegalInputMap().get(c);
            if ( input != null )
            {
              fillerMoveSet.add(input.getName());
            }
          }
        }

        LOGGER.info("Filler move set: " + fillerMoveSet);

        //  Now include all legals and goals for roles which already have ANY moves included
        //  For most games this will be all of them, but in games where the opponents moves do not
        //  impact anything on which our goal values are dependent they are irrelevant (canonical
        //  example is dual Hamilton)
        isPseudoPuzzle = true;

        for(Role role : pn.getRoles())
        {
          if ( role.equals(ourRole))
          {
            continue;
          }

          boolean fullyIncludeRole = false;

          for ( PolymorphicProposition p : pn.getLegalPropositions().get(role))
          {
            if ( reachableComponents.contains(p))
            {
              fullyIncludeRole = true;
              break;
            }
          }

          if ( fullyIncludeRole )
          {
            isPseudoPuzzle = false;
            //  Add in this role's goals and legals
            for( PolymorphicProposition c : pn.getGoalPropositions().get(role))
            {
              recursiveFindReachable(pn, c, reachableComponents, reachableComponents, null, true);
            }
            for( PolymorphicProposition c : pn.getLegalPropositions().get(role))
            {
              recursiveFindReachable(pn, c, reachableComponents, reachableComponents, null, true);

              //  If a legal is no within the core reachable set it has no impact on anything
              //  that influences our goals and is therefore only relevant as a possible source of
              //  what amounts to a noop (in the subset of relevant components)
              if ( !coreReachableComponents.contains(c) )
              {
                PolymorphicProposition input = pn.getLegalInputMap().get(c);
                if ( input != null )
                {
                  fillerMoveSet.add(input.getName());
                }
              }
            }
          }
          else
          {
            //  Arbitrarily keep one legal (it will be a noop, but we need one to make the
            //  game legal) and make sure it is always available
            PolymorphicProposition l = pn.getLegalPropositions().get(role)[0];

            assert(!reachableComponents.contains(l));
            reachableComponents.add(l);

            PolymorphicComponent oldInput = l.getSingleInput();
            oldInput.removeOutput(l);
            l.removeInput(oldInput);
            trueConstant.addOutput(l);
            l.addInput(trueConstant);

            //  Keep the lowest valued goal and set it unconditionally true
            int lowestVal = Integer.MAX_VALUE;
            PolymorphicProposition retainedGoalProp = null;

            for(PolymorphicProposition p : pn.getGoalPropositions().get(role))
            {
              int value = Integer.parseInt(p.getName().getBody().get(1).toString());
              if ( value < lowestVal )
              {
                lowestVal = value;
                retainedGoalProp = p;
              }
            }

            assert(retainedGoalProp != null);

            reachableComponents.add(retainedGoalProp);

            oldInput = retainedGoalProp.getSingleInput();
            oldInput.removeOutput(retainedGoalProp);
            retainedGoalProp.removeInput(oldInput);
            trueConstant.addOutput(retainedGoalProp);
            retainedGoalProp.addInput(trueConstant);
          }
        }
      }
      else
      {
        isPseudoPuzzle = true;
      }
    }
    else
    {
      //  Add all goals and legals if we have not been given a primary role
      for(PolymorphicProposition[] goals : pn.getGoalPropositions().values())
      {
        for(PolymorphicProposition c : goals)
        {
          recursiveFindReachable(pn, c, reachableComponents, null, null, true);
        }
      }

      for(PolymorphicProposition[] legals : pn.getLegalPropositions().values())
      {
        for(PolymorphicProposition c : legals)
        {
          recursiveFindReachable(pn, c, reachableComponents, null, null, true);
        }
      }
    }

    //  Anything potentially removable?
    boolean anyGoalsPotentiallyDiscardable = false;

    for(Role role : pn.getRoles())
    {
      for(PolymorphicProposition goalProp : pn.getGoalPropositions().get(role))
      {
        if ( !reachableComponents.contains(goalProp))
        {
          anyGoalsPotentiallyDiscardable = true;
          break;
        }
      }
    }
    if ( !reachableComponents.containsAll(pn.getBasePropositions().values()) ||
         !reachableComponents.containsAll(pn.getInputPropositions().values()) ||
         anyGoalsPotentiallyDiscardable)
    {
      //  Add in also any base props implied by definitely included does props
      //  which can impact included legals or other included base props
      Set<PolymorphicComponent> definatelyReachableDoesProps = new HashSet<>();
      for(PolymorphicComponent c : reachableComponents)
      {
        if (pn.getInputPropositions().values().contains(c))
        {
          definatelyReachableDoesProps.add(c);
        }
      }
      boolean anythingAdded;

      do
      {
        anythingAdded = false;

        Set<PolymorphicComponent> downstreamReachable = new HashSet<>();

        for(PolymorphicComponent c : definatelyReachableDoesProps)
        {
          anythingAdded |= addImpliedRequiredBaseProps(pn, c, reachableComponents, downstreamReachable);
        }
      } while(anythingAdded);

      //  What can we eliminate?
      Set<PolymorphicComponent> unreachable = new HashSet<>();
      int removalCount = 0;
      for(PolymorphicProposition c : pn.getBasePropositions().values())
      {
        if ( !reachableComponents.contains(c))
        {
          unreachable.add(c);
        }
      }

      for(PolymorphicProposition[] goals : pn.getGoalPropositions().values())
      {
        for(PolymorphicProposition c : goals)
        {
          if ( !reachableComponents.contains(c))
          {
            unreachable.add(c);
          }
        }
      }

      if ( pn.getRoles().length > 1 )
      {
        for( PolymorphicProposition[] legals : pn.getLegalPropositions().values())
        {
          for( PolymorphicProposition c : legals)
          {
            if ( !reachableComponents.contains(c))
            {
              unreachable.add(c);
            }
          }
        }
      }

      if ( !unreachable.isEmpty() )
      {
        PolymorphicConstant falseConst = pn.getComponentFactory().createConstant(-1, false);
        pn.addComponent(falseConst);

        for(PolymorphicComponent c : unreachable)
        {
          assert(c instanceof PolymorphicProposition);

          //  Replace with FALSE
          for(PolymorphicComponent output : c.getOutputs())
          {
            output.addInput(falseConst);
            falseConst.addOutput(output);
          }
          pn.removeComponent(c);
          removalCount++;
        }
      }

      LOGGER.debug("Removed " + removalCount + " irrelevant propositions");
    }

    return isPseudoPuzzle;
  }

  private static boolean addImpliedRequiredBaseProps(PolymorphicPropNet pn, PolymorphicComponent c, Set<PolymorphicComponent> reachableComponents, Set<PolymorphicComponent> newlyReachableComponents)
  {
    boolean result = false;

    if ( !newlyReachableComponents.contains(c))
    {
      newlyReachableComponents.add(c);

      if ( pn.getBasePropositions().values().contains(c) )
      {
        if ( !reachableComponents.contains(c))
        {
          if ( supportsRequiredComponent(c, reachableComponents) )
          {
            reachableComponents.add(c);
            result = true;
          }
        }
      }
      else
      {
        for(PolymorphicComponent output : c.getOutputs())
        {
          result |= addImpliedRequiredBaseProps(pn, output, reachableComponents, newlyReachableComponents);
        }
      }
    }

    return result;
  }

  private static boolean supportsRequiredComponent(PolymorphicComponent c, Set<PolymorphicComponent> reachableComponents)
  {
    for(PolymorphicComponent output : c.getOutputs())
    {
      if ( supportsRequiredComponentInternal(output, reachableComponents))
      {
        return true;
      }
    }

    return false;
  }

  private static boolean supportsRequiredComponentInternal(PolymorphicComponent c, Set<PolymorphicComponent> reachableComponents)
  {
    if ( c instanceof PolymorphicProposition )
    {
      //  Legal or base - if already included return true else false
      return reachableComponents.contains(c);
    }

    for(PolymorphicComponent output : c.getOutputs())
    {
      if ( supportsRequiredComponentInternal(output, reachableComponents))
      {
        return true;
      }
    }

    return false;
  }

  private static Boolean requiresDoes(PolymorphicComponent c)
  {
    if ( c instanceof PolymorphicProposition && ((PolymorphicProposition)c).getName().getName() == GdlPool.DOES )
    {
      return true;
    }
    else if ( c instanceof PolymorphicTransition)
    {
      return false;
    }
    else if ( c instanceof PolymorphicAnd )
    {
      for(PolymorphicComponent input: c.getInputs())
      {
        if (requiresDoes(input))
        {
          return true;
        }
      }

      return false;
    }
    else if ( c instanceof PolymorphicOr )
    {
      for(PolymorphicComponent input: c.getInputs())
      {
        if (!requiresDoes(input))
        {
          return false;
        }
      }

      return true;
    }
    else if ( c instanceof PolymorphicNot )
    {
      //  Give up if there is inversion involved - gets too complex and this is only
      //  to optimize trimming - erring on the side of returning false is always
      //  functionally safe
      return false;
    }
    else
    {
      if ( c.getInputs().isEmpty() )
      {
        return false;
      }

      return requiresDoes(c.getSingleInput());
    }
  }

  /**
   * Recursively walk backwards through the propnet to find reachable components.
   * If an assertedReachable set is provided then anything in that set is considered
   * even if the self-support logic would otherwise treat it as not reachable (based on
   * reasoning that if it is not already true the self-support loop cannot make it true)
   * @param pn
   * @param from
   * @param assertedReachableComponents
   * @param reachableComponents
   * @param viaBaseProp
   * @param traverseMoves if true work back through does->legal linkages else don't
   */
  private static void recursiveFindReachable(PolymorphicPropNet pn,
                                             PolymorphicComponent from,
                                             Set<PolymorphicComponent> reachableComponents,
                                             Set<PolymorphicComponent> assertedReachableComponents,
                                             PolymorphicProposition viaBaseProp,
                                             Boolean traverseMoves)
  {
    if (reachableComponents.contains(from))
    {
      return;
    }

    if (pn.getBasePropositions().values().contains(from))
    {
      if ( assertedReachableComponents != null )
      {
        viaBaseProp = (PolymorphicProposition)from;

        // viaBaseProp is used to prune parts of the network which can't possibly cause a proposition to become true.
        // However, if the proposition is initially true, we should avoid that pruning.  Also if it is asserted to
        // be reachable we should similarly avoid trimming it - in usage any base proposition on which a goal is dependent
        // without crossing any move boundaries (so directly dependent in that state) is asserted reachable in this way
        if (assertedReachableComponents.contains(viaBaseProp))
        {
          viaBaseProp = null;
        }
      }
    }
    else if ( pn.getLegalInputMap().containsKey(from) && ((PolymorphicProposition)from).getName().getName() == GdlPool.DOES )
    {
      if ( traverseMoves )
      {
        reachableComponents.add(from);
        recursiveFindReachable(pn, pn.getLegalInputMap().get(from), reachableComponents, assertedReachableComponents, null, traverseMoves);
      }

      return;
    }
    else if (from instanceof PolymorphicAnd)
    {
      if ( viaBaseProp != null )
      {
        //  Don't traverse conditions that just preserve an already set base prop, since
        //  such a path cannot CAUSE it to become set
        if (from.getInputs().contains(viaBaseProp))
        {
          return;
        }
      }
      else if ( !traverseMoves )
      {
        //  If we're not traversing moves also don't traverse logic that requires a move to be active.
        //  For now we just handle the case of all branches directly leading (eventually) only to DOES props
        for(PolymorphicComponent c: from.getInputs())
        {
          if ( requiresDoes(c))
          {
            return;
          }
        }
      }
    }

    reachableComponents.add(from);

    for(PolymorphicComponent c : from.getInputs())
    {
      // When StackOverflowError has been seen here previously, it has turned out to be a stack size issue (rather
      // than an infinite recursion bug).  On Windows, with 64-bit Oracle JVM, 2MB stack is big enough.  (1MB isn't.)
      // To configure, use -Xss2m.
      recursiveFindReachable(pn, c, reachableComponents, assertedReachableComponents, viaBaseProp, traverseMoves);
    }
  }

  public static void removeUnreachableBasesAndInputs(PolymorphicPropNet pn)
  {
    removeUnreachableBasesAndInputs(pn, false);
  }

  public static void removeUnreachableBasesAndInputs(PolymorphicPropNet pn,
                                                     boolean preserveAllTransitions)
  {
    PolymorphicComponentFactory componentFactory = pn.getComponentFactory();
    //This actually might remove more than bases and inputs
    //We flow through the game graph to see what can be true (and what can be false?)...
    Map<PolymorphicComponent, Type> reachability = new HashMap<>();
    Set<GdlTerm> initted = new HashSet<>();
    for (PolymorphicComponent c : pn.getComponents())
    {
      reachability.put(c, Type.STAR);
      if (c instanceof PolymorphicProposition)
      {
        PolymorphicProposition p = (PolymorphicProposition)c;
        if (p.getName() instanceof GdlRelation)
        {
          GdlRelation r = (GdlRelation)p.getName();
          if (r.getName().equals(GdlPool.INIT))
          {
            //Add the base
            initted.add(r.get(0));
          }
        }
      }
      else if (preserveAllTransitions && c instanceof PolymorphicTransition)
      {
        reachability.put(c, Type.BOTH);
      }
    }

    //Set<Component> toReevaluate = new LinkedHashSet<Component>();
    ReEvaulationSet toReevaluate = new ReEvaulationSet(reachability);
    Set<PolymorphicComponent> explicitlyInitedBases = new HashSet<>();

    for (PolymorphicComponent c : pn.getComponents())
    {
      //	Descendents of TRUE constants must be considered
      if (c instanceof PolymorphicConstant && c.getValue())
      {
        toReevaluate.addAll(c.getOutputs());
      }
      //	If we want to preserve transitions regardless of their inputs
      //	tag them as able to take any vlaue and force reconsideration of their
      //	children
      else if (preserveAllTransitions && c instanceof PolymorphicTransition)
      {
        reachability.put(c, Type.BOTH);
        toReevaluate.addAll(c.getOutputs());
      }
    }

    //Every input can be false (we assume that no player will have just one move allowed all game)
    for (PolymorphicProposition p : pn.getInputPropositions().values())
    {
      if (pn.getLegalInputMap().containsKey(p))
      {
        reachability.put(p, Type.BOTH);
      }
      else
      {
        reachability.put(p, Type.FALSE);
      }
      toReevaluate.addAll(p.getOutputs());
    }
    //Every base with "init" can be true, every base without "init" can be false
    for (Entry<GdlSentence, PolymorphicProposition> entry : pn
        .getBasePropositions().entrySet())
    {
      PolymorphicProposition p = entry.getValue();
      //So, does it have init?
      //TODO: Remove "true" dereferencing? Need "global" option for that
      //if(initted.contains(entry.getKey())) {
      if (entry.getKey() instanceof GdlRelation &&
          initted.contains(((GdlRelation)entry.getKey()).get(0)))
      {
        reachability.put(p, Type.TRUE);
        explicitlyInitedBases.add(p);
      }
      else
      {
        reachability.put(entry.getValue(), Type.FALSE);
      }
      toReevaluate.addAll(p.getOutputs());
    }
    //Might as well add in INIT
    PolymorphicProposition initProposition = pn.getInitProposition();
    if (initProposition != null)
    {
      reachability.put(initProposition, Type.BOTH);
      toReevaluate.addAll(initProposition.getOutputs());
    }

    int loopDetectionCount = 0;
    int reEvaluationLimit = toReevaluate.size();

    //Now, propagate everything we know
    while (!toReevaluate.isEmpty())
    {
      if (loopDetectionCount++ > reEvaluationLimit)
      {
        //	We're in a loop of undecided components - just leave anything that's still undecided untrimmed
        //	For now we take a REALLY simplistic approach and promote everything caught in the looping to
        //	BOTH
        //Set<Component> newReEvaulateList = new LinkedHashSet<Component>();
        ReEvaulationSet newReEvaulateList = new ReEvaulationSet(reachability);

        for (PolymorphicComponent curComp : toReevaluate)
        {
          Type type = reachability.get(curComp);
          type = addTrue(type);
          type = addFalse(type);
          reachability.put(curComp, type);

          newReEvaulateList.addAll(curComp.getOutputs());
        }

        toReevaluate = newReEvaulateList;
        loopDetectionCount = 0;
        reEvaluationLimit = toReevaluate.size();
        continue;
      }
      PolymorphicComponent curComp = toReevaluate.iterator().next();
      toReevaluate.remove(curComp);
      //Can we upgrade its type?
      Type type = reachability.get(curComp);
      boolean checkTrue = true, checkFalse = true;
      if (type == Type.BOTH)
      { //Nope
        continue;
      }
      else if (type == Type.TRUE)
      {
        checkTrue = false;
      }
      else if (type == Type.FALSE)
      {
        checkFalse = false;
      }
      boolean upgradeTrue = false, upgradeFalse = false;
      boolean curCompIsLegalProposition = false;

      //How we react to the parents (or pseudo-parents) depends on the type
      Collection<? extends PolymorphicComponent> parents = curComp.getInputs();
      if (curComp instanceof PolymorphicAnd)
      {
        if (checkTrue)
        {
          //All parents must be capable of being true
          boolean allCanBeTrue = true;
          Set<PolymorphicComponent> starParents = new HashSet<>();
          for (PolymorphicComponent parent : parents)
          {
            Type parentType = reachability.get(parent);
            if (parent instanceof PolymorphicConstant)
            {
              parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
            }
            if (parentType == Type.STAR)
            {
              starParents.add(parent);
            }
            else if (!parentType.hasTrue())
            {
              allCanBeTrue = false;
              break;
            }
          }
          if (starParents.size() > 0)
          {
            //	Cannot decide yet - add STAR parents for re-evaluation
            //	and then ourselves again
            toReevaluate.addAll(starParents);
            continue;
          }
          upgradeTrue = allCanBeTrue;
        }
        if (checkFalse)
        {
          //Some parent must be capable of being false
          Set<PolymorphicComponent> starParents = new HashSet<>();
          for (PolymorphicComponent parent : parents)
          {
            Type parentType = reachability.get(parent);
            if (parent instanceof PolymorphicConstant)
            {
              parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
            }
            if (parentType == Type.STAR)
            {
              starParents.add(parent);
            }
            else if (parentType.hasFalse())
            {
              upgradeFalse = true;
              break;
            }
          }
          if (!upgradeFalse && starParents.size() > 0)
          {
            //	Cannot decide yet - add STAR parents for re-evaluation
            //	and then ourselves again
            toReevaluate.addAll(starParents);
            continue;
          }
        }
      }
      else if (curComp instanceof PolymorphicOr)
      {
        if (checkTrue)
        {
          //Some parent must be capable of being true
          Set<PolymorphicComponent> starParents = new HashSet<>();
          for (PolymorphicComponent parent : parents)
          {
            Type parentType = reachability.get(parent);
            if (parent instanceof PolymorphicConstant)
            {
              parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
            }
            if (parentType == Type.STAR)
            {
              starParents.add(parent);
            }
            else if (parentType.hasTrue())
            {
              upgradeTrue = true;
              break;
            }
          }
          if (!upgradeTrue && starParents.size() > 0)
          {
            //	Cannot decide yet - add STAR parents for re-evaluation
            //	and then ourselves again
            toReevaluate.addAll(starParents);
            continue;
          }
        }
        if (checkFalse)
        {
          //All parents must be capable of being false
          boolean allCanBeFalse = true;
          Set<PolymorphicComponent> starParents = new HashSet<>();
          for (PolymorphicComponent parent : parents)
          {
            Type parentType = reachability.get(parent);
            if (parent instanceof PolymorphicConstant)
            {
              parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
            }
            if (parentType == Type.STAR)
            {
              starParents.add(parent);
            }
            else if (!parentType.hasFalse())
            {
              allCanBeFalse = false;
              break;
            }
          }
          if (starParents.size() > 0)
          {
            //	Cannot decide yet - add STAR parents for re-evaluation
            //	and then ourselves again
            toReevaluate.addAll(starParents);
            continue;
          }
          upgradeFalse = allCanBeFalse;
        }
      }
      else if (curComp instanceof PolymorphicNot)
      {
        PolymorphicComponent parent = curComp.getSingleInput();
        Type parentType = reachability.get(parent);
        if (parent instanceof PolymorphicConstant)
        {
          parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
        }
        if (parentType == Type.STAR)
        {
          toReevaluate.add(parent);
          continue;
        }
        if (checkTrue && parentType.hasFalse())
          upgradeTrue = true;
        if (checkFalse && parentType.hasTrue())
          upgradeFalse = true;
      }
      else if (curComp instanceof PolymorphicTransition)
      {
        PolymorphicComponent parent = curComp.getSingleInput();
        Type parentType = reachability.get(parent);
        if (parent instanceof PolymorphicConstant)
        {
          parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
        }
        if (parentType == Type.STAR)
        {
          toReevaluate.add(parent);
          continue;
        }
        if (checkTrue && parentType.hasTrue())
          upgradeTrue = true;
        if (checkFalse && parentType.hasFalse())
          upgradeFalse = true;
      }
      else if (curComp instanceof PolymorphicProposition)
      {
        //TODO: Special case: Inputs
        PolymorphicProposition p = (PolymorphicProposition)curComp;
        if (pn.getLegalInputMap().containsKey(curComp))
        {
          GdlRelation r = (GdlRelation)p.getName();
          if (r.getName().equals(GdlPool.DOES))
          {
            //The legal prop. is a pseudo-parent
            PolymorphicComponent legal = pn.getLegalInputMap().get(curComp);
            Type legalType = reachability.get(legal);
            if (checkTrue && legalType.hasTrue())
              upgradeTrue = true;
            if (checkFalse && legalType.hasFalse())
              upgradeFalse = true;
          }
          else
          {
            curCompIsLegalProposition = true;
          }
        }

        //Otherwise, just do same as Transition... easy
        if (curComp.getInputs().size() == 1)
        {
          PolymorphicComponent parent = curComp.getSingleInput();
          Type parentType = reachability.get(parent);
          if (parent instanceof PolymorphicConstant)
          {
            parentType = parent.getValue() ? Type.TRUE : Type.FALSE;
          }
          if (parentType == Type.STAR)
          {
            toReevaluate.add(parent);
            continue;
          }
          if (checkTrue && parentType.hasTrue())
            upgradeTrue = true;
          if (checkFalse && parentType.hasFalse())
            upgradeFalse = true;
        }
      }
      else
      {
        //Constants won't get added
        throw new RuntimeException("Unexpected node type " +
                                   curComp.getClass());
      }

      //Deal with upgrades
      if (upgradeTrue)
      {
        type = addTrue(type);
        reachability.put(curComp, type);
      }
      if (upgradeFalse)
      {
        type = addFalse(type);
        reachability.put(curComp, type);
      }
      if (upgradeTrue || upgradeFalse)
      {
        toReevaluate.addAll(curComp.getOutputs());
        //Don't forget: if "legal", check "does"
        if (curCompIsLegalProposition)
        {
          PolymorphicProposition input = pn.getLegalInputMap().get(curComp);
          toReevaluate.add(input);
        }
        loopDetectionCount = 0;
        reEvaluationLimit = toReevaluate.size();
      }
    }

    //We deliberately shouldn't remove the stuff attached to TRUE... or anything that's
    //always true...
    //But we should be able to remove bases and inputs (when it's justified)

    //TODO: Go through all the cases of everything I can dump
    //For now... how about inputs?
    PolymorphicConstant trueConst = componentFactory.createConstant(-1, true);
    PolymorphicConstant alternateTrueConst = componentFactory
        .createConstant(-1, true);
    PolymorphicConstant falseConst = componentFactory
        .createConstant(-1, false);
    pn.addComponent(trueConst);
    pn.addComponent(alternateTrueConst); //	Used for nodes we need to keep but which are always true
    pn.addComponent(falseConst);
    //Make them the input of all false/true components
    for (Entry<PolymorphicComponent, Type> entry : reachability.entrySet())
    {
      Type type = entry.getValue();
      if (type == Type.TRUE || type == Type.FALSE)
      {
        PolymorphicComponent c = entry.getKey();
        //Disconnect from inputs
        for (PolymorphicComponent input : c.getInputs())
        {
          input.removeOutput(c);
        }
        c.removeAllInputs();
        if ((type == Type.TRUE) != (c instanceof PolymorphicNot))
        {
          if (explicitlyInitedBases.contains(c))
          {
            //	Cannot dump bases that are init'd else they'll be missing from the state
            //	in which they should appear essentially as a fixture - hook them up to
            //	a constant TRUE that we will not trim descendants of
            c.addInput(alternateTrueConst);
            alternateTrueConst.addOutput(c);
          }
          else
          {
            c.addInput(trueConst);
            trueConst.addOutput(c);
          }
        }
        else
        {
          c.addInput(falseConst);
          falseConst.addOutput(c);
        }
      }
    }

    //then...
    //optimizeAwayTrueAndFalse(null, null, trueConst, falseConst);
    optimizeAwayTrueAndFalse(pn, trueConst, falseConst);

  }

  private static Type addTrue(Type type)
  {
    switch (type)
    {
      case STAR:
        return Type.TRUE;
      case TRUE:
        return Type.TRUE;
      case FALSE:
        return Type.BOTH;
      case BOTH:
        return Type.BOTH;
      default:
        throw new RuntimeException("Unanticipated node type " + type);
    }
  }

  private static Type addFalse(Type type)
  {
    switch (type)
    {
      case STAR:
        return Type.FALSE;
      case TRUE:
        return Type.BOTH;
      case FALSE:
        return Type.FALSE;
      case BOTH:
        return Type.BOTH;
      default:
        throw new RuntimeException("Unanticipated node type " + type);
    }
  }

  /**
   * Optimizes an already-existing propnet by removing useless leaves. These
   * are components that have no outputs, but have no special meaning in GDL
   * that requires them to stay. TODO: Currently fails on propnets with cycles.
   *
   * @param pn
   */
  public static void lopUselessLeaves(PropNet pn)
  {
    //Approach: Collect useful propositions based on a backwards
    //search from goal/legal/terminal (passing through transitions)
    Set<Component> usefulComponents = new HashSet<>();
    //TODO: Also try with queue?
    Stack<Component> toAdd = new Stack<>();
    toAdd.add(pn.getTerminalProposition());
    usefulComponents.add(pn.getInitProposition()); //Can't remove it...
    for (Set<Proposition> goalProps : pn.getGoalPropositions().values())
      toAdd.addAll(goalProps);
    for (Set<Proposition> legalProps : pn.getLegalPropositions().values())
      toAdd.addAll(legalProps);
    while (!toAdd.isEmpty())
    {
      Component curComp = toAdd.pop();
      if (usefulComponents.contains(curComp))
        //We've already added it
        continue;
      usefulComponents.add(curComp);
      toAdd.addAll(curComp.getInputs());
    }

    //Remove the components not marked as useful
    List<Component> allComponents = new ArrayList<>(pn.getComponents());
    for (Component c : allComponents)
    {
      if (!usefulComponents.contains(c))
        pn.removeComponent(c);
    }
  }

  /**
   * Optimizes an already-existing propnet by removing propositions of the form
   * (init ?x). Does NOT remove the proposition "INIT".
   *
   * @param pn
   */
  public static void removeInits(PropNet pn)
  {
    List<Proposition> toRemove = new ArrayList<>();
    for (Proposition p : pn.getPropositions())
    {
      if (p.getName() instanceof GdlRelation)
      {
        GdlRelation relation = (GdlRelation)p.getName();
        if (relation.getName() == GdlPool.INIT)
        {
          toRemove.add(p);
        }
      }
    }

    for (Proposition p : toRemove)
    {
      pn.removeComponent(p);
    }
  }

  /**
   * Simplify the network by removal of constants and gates which are redundant.
   *
   * @param pn - the propnet to modify.
   * @param allowRemovalOfInputProps - whether input propositions can be removed (if found to be redundant).
   */
  public static void removeRedundantConstantsAndGates(PolymorphicPropNet pn,
                                                      boolean allowRemovalOfInputProps)
  {
    //	What about constants other than true and false props?
    Set<PolymorphicComponent> redundantComponents = new HashSet<>();
    int removedRedundantComponentCount = 0;

    PolymorphicComponent trueConst = pn.getComponentFactory().createConstant(-1, true);
    PolymorphicComponent falseConst = pn.getComponentFactory().createConstant(-1, false);

    assert(trueConst != null);
    assert(falseConst != null);

    pn.addComponent(trueConst);
    pn.addComponent(falseConst);

    do
    {
      redundantComponents.clear();

      //  First eliminate logic hard-gated by constant inputs
      for (PolymorphicComponent c : pn.getComponents())
      {
        if ( c instanceof PolymorphicConstant )
        {
          boolean isTrue = c.getValue();

          Set<PolymorphicComponent> newOutputs = new HashSet<>();
          for(PolymorphicComponent output : c.getOutputs())
          {
            if ( (!isTrue && (output instanceof PolymorphicAnd)) ||
                 (isTrue && (output instanceof PolymorphicOr)) )
            {
              //  Remove all input to the following component apart from the constant
              //  the next stage will then remove it as redundant
              for(PolymorphicComponent input : output.getInputs())
              {
                if ( input != c )
                {
                  input.removeOutput(output);
                }
              }

              output.removeAllInputs();
              output.addInput(c);
            }
            else if ( output instanceof PolymorphicProposition )
            {
              //  Fixed value proposition - needed, but its outputs can be connected
              //  directly to the source constant rather than the prop, and hence may be eliminatable
              newOutputs.addAll(output.getOutputs());
              for(PolymorphicComponent downstream : output.getOutputs())
              {
                downstream.removeInput(output);
                newOutputs.add(downstream);
              }
              output.removeAllOutputs();
            }
            else if (output instanceof PolymorphicNot)
            {
              // Replace TRUE -> NOT with FALSE for all downstream components.
              // Replace FALSE -> NOT with TRUE for all downstream components.
              for(PolymorphicComponent downstream : output.getOutputs())
              {
                PolymorphicComponent replacement = isTrue ? falseConst : trueConst;
                downstream.removeInput(output);
                downstream.addInput(replacement);
                replacement.addOutput(downstream);
              }
              output.removeAllOutputs();
            }
          }

          for(PolymorphicComponent downstream : newOutputs)
          {
            c.addOutput(downstream);
            downstream.addInput(c);
          }
        }
      }

      for (PolymorphicComponent c : pn.getComponents())
      {
        if (c instanceof PolymorphicConstant)
        {
          if (c != trueConst &&
              c != falseConst)
          {
            redundantComponents.add(c);
          }
        }
        else if (c instanceof PolymorphicAnd)
        {
          if (c.getInputs().size() < 2 ||
              c.getOutputs().isEmpty() ||
              (c.getOutputs().size() == 1 && c.getSingleOutput() instanceof PolymorphicAnd))
          {
            redundantComponents.add(c);
          }
        }
        else if (c instanceof PolymorphicOr)
        {
          if (c.getInputs().size() < 2 ||
              c.getOutputs().isEmpty() ||
              (c.getOutputs().size() == 1 && c.getSingleOutput() instanceof PolymorphicOr))
          {
            redundantComponents.add(c);
          }
        }
        else if (c instanceof PolymorphicNot)
        {
          if (c.getInputs().isEmpty() ||
              c.getOutputs().isEmpty() ||
              (c.getOutputs().size() == 1 && c.getSingleOutput() instanceof PolymorphicNot))
          {
            redundantComponents.add(c);
          }
        }
      }

      for (PolymorphicComponent c : redundantComponents)
      {
        if ((c.getInputs().isEmpty() || c.getOutputs().isEmpty()) &&
            (c instanceof PolymorphicAnd || c instanceof PolymorphicOr || c instanceof PolymorphicNot))
        {
          //	Nothing further to do
        }
        else if (c instanceof PolymorphicConstant)
        {
          for (PolymorphicComponent output : c.getOutputs())
          {
            output.removeInput(c);
            if (c.getValue())
            {
              trueConst.addOutput(output);
              output.addInput(trueConst);
            }
            else
            {
              falseConst.addOutput(output);
              output.addInput(falseConst);
            }
          }
        }
        else if ((c instanceof PolymorphicAnd) || (c instanceof PolymorphicOr))
        {
          if ( c.getInputs().size() == 1 )
          {
            PolymorphicComponent input = c.getSingleInput();

            input.removeOutput(c);

            for (PolymorphicComponent output : c.getOutputs())
            {
              output.removeInput(c);
              input.addOutput(output);
              output.addInput(input);
            }
          }
          else
          {
            PolymorphicComponent output = c.getSingleOutput();

            output.removeInput(c);

            for (PolymorphicComponent input : c.getInputs())
            {
              input.removeOutput(c);
              input.addOutput(output);
              output.addInput(input);
            }
          }
        }
        else if (c instanceof PolymorphicNot)
        {
          PolymorphicComponent nextInLine = c.getSingleOutput();
          PolymorphicComponent input = c.getSingleInput();

          input.removeOutput(c);

          for (PolymorphicComponent output : nextInLine.getOutputs())
          {
            output.removeInput(nextInLine);
            output.addInput(input);
            input.addOutput(output);
          }

          removedRedundantComponentCount++;
          pn.removeComponent(nextInLine);
        }

        removedRedundantComponentCount++;
        pn.removeComponent(c);
      }
    }
    while (redundantComponents.size() > 0);

    // Remove the following redundant inputs.
    // - TRUE inputs to ANDs
    // - FALSE inputs to ORs
    // - Single input ANDs/ORs
    List<PolymorphicComponent> eliminations = new LinkedList<>();

    do
    {
      eliminations.clear();

      if (!trueConst.getOutputs().isEmpty())
      {
        List<PolymorphicComponent> disconnected = new LinkedList<>();

        for (PolymorphicComponent c : trueConst.getOutputs())
        {
          if (c instanceof PolymorphicAnd && c.getInputs().size() > 1)
          {
            c.removeInput(trueConst);
            disconnected.add(c);
          }
        }
        for (PolymorphicComponent c : disconnected)
        {
          trueConst.removeOutput(c);
        }
      }

      if (!falseConst.getOutputs().isEmpty())
      {
        List<PolymorphicComponent> disconnected = new LinkedList<>();

        for (PolymorphicComponent c : falseConst.getOutputs())
        {
          if (c instanceof PolymorphicOr && c.getInputs().size() > 1)
          {
            c.removeInput(falseConst);
            disconnected.add(c);
          }
          else if (c instanceof PolymorphicProposition &&
                   !isGoalProp((PolymorphicProposition)c) &&
                   c.getOutputs().size() == 0)
          {
            c.removeInput(falseConst);
            disconnected.add(c);
          }
        }
        for (PolymorphicComponent c : disconnected)
        {
          falseConst.removeOutput(c);
        }
      }


      for (PolymorphicComponent c : pn.getComponents())
      {
        if (c instanceof PolymorphicAnd || c instanceof PolymorphicOr)
        {
          if (c.getInputs().size() == 1)
          {
            PolymorphicComponent input = c.getSingleInput();

            for (PolymorphicComponent output : c.getOutputs())
            {
              output.removeInput(c);
              output.addInput(input);
              input.addOutput(output);
            }

            eliminations.add(c);
          }
          else if (c.getOutputs().size() == 0)
          {
            eliminations.add(c);
          }
        }
        else if (c.getOutputs().size() == 0)
        {
          if (!isEssentialProposition(c, c.getInputs().size() == 0) &&
              (!(c instanceof PolymorphicProposition) || !pn
                  .getBasePropositions().values()
                  .contains(c)))
          {
            if (allowRemovalOfInputProps ||
                !pn.getInputPropositions().values().contains(c))
            {
              eliminations.add(c);
            }
          }
        }
      }

      for (PolymorphicComponent c : eliminations)
      {
        for (PolymorphicComponent input : c.getInputs())
        {
          input.removeOutput(c);
        }

        pn.removeComponent(c);
        removedRedundantComponentCount++;
      }
    }
    while (eliminations.size() > 0);

    LOGGER.debug("Removed " + removedRedundantComponentCount + " redundant components");
  }

  private static boolean isGoalProp(PolymorphicProposition p)
  {
    GdlSentence s = p.getName();

    if ( s instanceof GdlRelation )
    {
      return s.getName().equals(GdlPool.GOAL);
    }

    return false;
  }

  private static final int largeGateThreshold = 5;

  public static void refactorLargeGates(PolymorphicPropNet pn)
  {
    Set<PolymorphicComponent> newComponents = new HashSet<>();
    Set<PolymorphicComponent> removedComponents = new HashSet<>();
    int inputToOutputFactorizationRemovedCount = 0;
    int inputToOutputFactorizationAddedCount = 0;

    for (PolymorphicComponent c : pn.getComponents())
    {
      if (c.getInputs().size() > largeGateThreshold)
      {
        if ((c instanceof PolymorphicOr))
        {
          //	Can we find a common factor across input ANDs?
          Set<PolymorphicComponent> inputANDinputs = new HashSet<>();
          boolean nonAndsPresent = false;
          int numFactorInstances = 0;
          int numPotentiallyRemovableComponents = 0;

          for (PolymorphicComponent input : c.getInputs())
          {
            if (!(input instanceof PolymorphicAnd))
            {
              //	Not uniform
              nonAndsPresent = true;
              continue;
            }

            if (input.getOutputs().size() != 1)
            {
              //	If the output is used for other purposes too we cannot factor it
              inputANDinputs.clear();
              break;
            }

            numFactorInstances++;

            if (inputANDinputs.isEmpty())
            {
              inputANDinputs.addAll(input.getInputs());
            }
            else
            {
              Collection<? extends PolymorphicComponent> testInputs = input
                  .getInputs();

              for (Iterator<PolymorphicComponent> itr = inputANDinputs
                  .iterator(); itr.hasNext();)
              {
                if (!testInputs.contains(itr.next()))
                {
                  itr.remove();
                }
              }

              if (inputANDinputs.isEmpty())
              {
                break;
              }

              if (testInputs.size() <= inputANDinputs.size() + 1)
              {
                numPotentiallyRemovableComponents++;
              }
            }
          }

          if (!inputANDinputs.isEmpty() && numFactorInstances > 1 &&
              numPotentiallyRemovableComponents > 1)
          {
            PolymorphicComponent factoredOr;

            if (nonAndsPresent)
            {
              factoredOr = pn.getComponentFactory().createOr(-1, -1);
              newComponents.add(factoredOr);
              inputToOutputFactorizationAddedCount++;
            }
            else
            {
              factoredOr = c;
            }

            Set<PolymorphicComponent> removedOrInputs = new HashSet<>();
            Set<PolymorphicComponent> addedOrInputs = new HashSet<>();

            //	We have one or more common factors - move them past the OR
            for (PolymorphicComponent input : c.getInputs())
            {
              if (input instanceof PolymorphicAnd)
              {
                for (PolymorphicComponent factoredInput : inputANDinputs)
                {
                  input.removeInput(factoredInput);
                  factoredInput.removeOutput(input);
                }

                switch (input.getInputs().size())
                {
                  case 0:
                    removedComponents.add(input);
                    removedOrInputs.add(input);
                    inputToOutputFactorizationRemovedCount++;
                    break;
                  case 1:
                    removedComponents.add(input);
                    removedOrInputs.add(input);
                    inputToOutputFactorizationRemovedCount++;

                    PolymorphicComponent source = input.getSingleInput();

                    if (c == factoredOr)
                    {
                      addedOrInputs.add(source);
                    }
                    else
                    {
                      factoredOr.addInput(source);
                    }
                    source.removeOutput(input);
                    source.addOutput(factoredOr);
                    break;
                  default:
                    if (c != factoredOr)
                    {
                      removedOrInputs.add(input);
                      input.removeOutput(c);
                      factoredOr.addInput(input);
                      input.addOutput(factoredOr);
                    }
                    break;
                }
              }
            }

            for (PolymorphicComponent removed : removedOrInputs)
            {
              c.removeInput(removed);
            }

            for (PolymorphicComponent added : addedOrInputs)
            {
              c.addInput(added);
            }

            PolymorphicAnd newAnd = pn.getComponentFactory().createAnd(-1, -1);
            newComponents.add(newAnd);
            inputToOutputFactorizationAddedCount++;

            if (c == factoredOr)
            {
              for (PolymorphicComponent output : c.getOutputs())
              {
                newAnd.addOutput(output);
                output.removeInput(c);
                output.addInput(newAnd);
              }

              c.removeAllOutputs();
            }
            else
            {
              newAnd.addOutput(c);
              c.addInput(newAnd);
            }

            factoredOr.addOutput(newAnd);
            newAnd.addInput(factoredOr);

            for (PolymorphicComponent input : inputANDinputs)
            {
              input.addOutput(newAnd);
              newAnd.addInput(input);
            }
          }
        }
        else if ((c instanceof PolymorphicAnd))
        {
          //	Can we find a common factor across input ORs?
          Set<PolymorphicComponent> inputORinputs = new HashSet<>();
          boolean nonOrsPresent = false;
          int numFactorInstances = 0;
          int numPotentiallyRemovableComponents = 0;

          for (PolymorphicComponent input : c.getInputs())
          {
            if (!(input instanceof PolymorphicOr))
            {
              //	Not uniform
              nonOrsPresent = true;
              continue;
            }

            if (input.getOutputs().size() != 1)
            {
              //	If the output is used for other purposes too we cannot factor it
              inputORinputs.clear();
              break;
            }

            numFactorInstances++;

            if (inputORinputs.isEmpty())
            {
              inputORinputs.addAll(input.getInputs());
            }
            else
            {
              Collection<? extends PolymorphicComponent> testInputs = input
                  .getInputs();

              for (Iterator<PolymorphicComponent> itr = inputORinputs
                  .iterator(); itr.hasNext();)
              {
                if (!testInputs.contains(itr.next()))
                {
                  itr.remove();
                }
              }

              if (inputORinputs.isEmpty())
              {
                break;
              }

              if (testInputs.size() <= inputORinputs.size() + 1)
              {
                numPotentiallyRemovableComponents++;
              }
            }
          }

          if (!inputORinputs.isEmpty() && numFactorInstances > 1 &&
              numPotentiallyRemovableComponents > 1)
          {
            PolymorphicComponent factoredAnd;

            if (nonOrsPresent)
            {
              factoredAnd = pn.getComponentFactory().createAnd(-1, -1);
              newComponents.add(factoredAnd);
              inputToOutputFactorizationAddedCount++;
            }
            else
            {
              factoredAnd = c;
            }

            Set<PolymorphicComponent> removedAndInputs = new HashSet<>();
            Set<PolymorphicComponent> addedAndInputs = new HashSet<>();

            //	We have one or more common factors - move them past the OR
            for (PolymorphicComponent input : c.getInputs())
            {
              if (input instanceof PolymorphicOr)
              {
                for (PolymorphicComponent factoredInput : inputORinputs)
                {
                  input.removeInput(factoredInput);
                  factoredInput.removeOutput(input);
                }

                switch (input.getInputs().size())
                {
                  case 0:
                    removedComponents.add(input);
                    removedAndInputs.add(input);
                    inputToOutputFactorizationRemovedCount++;
                    break;
                  case 1:
                    removedComponents.add(input);
                    removedAndInputs.add(input);
                    inputToOutputFactorizationRemovedCount++;

                    PolymorphicComponent source = input.getSingleInput();

                    if (c == factoredAnd)
                    {
                      addedAndInputs.add(source);
                    }
                    else
                    {
                      factoredAnd.addInput(source);
                    }
                    source.removeOutput(input);
                    source.addOutput(factoredAnd);
                    break;
                  default:
                    if (c != factoredAnd)
                    {
                      removedAndInputs.add(input);
                      input.removeOutput(c);
                      factoredAnd.addInput(input);
                      input.addOutput(factoredAnd);
                    }
                    break;
                }
              }
            }

            for (PolymorphicComponent removed : removedAndInputs)
            {
              c.removeInput(removed);
            }

            for (PolymorphicComponent added : addedAndInputs)
            {
              c.addInput(added);
            }

            PolymorphicOr newOr = pn.getComponentFactory().createOr(-1, -1);
            newComponents.add(newOr);
            inputToOutputFactorizationAddedCount++;

            if (c == factoredAnd)
            {
              for (PolymorphicComponent output : c.getOutputs())
              {
                newOr.addOutput(output);
                output.removeInput(c);
                output.addInput(newOr);
              }

              c.removeAllOutputs();
            }
            else
            {
              newOr.addOutput(c);
              c.addInput(newOr);
            }

            factoredAnd.addOutput(newOr);
            newOr.addInput(factoredAnd);

            for (PolymorphicComponent input : inputORinputs)
            {
              input.addOutput(newOr);
              newOr.addInput(input);
            }
          }
        }
      }
    }

    for (PolymorphicComponent c : newComponents)
    {
      pn.addComponent(c);
    }

    for (PolymorphicComponent c : removedComponents)
    {
      pn.removeComponent(c);
    }

    int inputToOutputFactorizationNetRemovedCount = inputToOutputFactorizationRemovedCount -
                                                    inputToOutputFactorizationAddedCount;
    LOGGER.debug("Net components removed by input->output factorization: " + inputToOutputFactorizationNetRemovedCount);
    if (inputToOutputFactorizationNetRemovedCount < 0)
    {
      LOGGER.warn("Unexpected growth in size from factorization");
    }
  }

  public static void refactorLargeFanouts(PolymorphicPropNet pn)
  {
    Set<PolymorphicComponent> newComponents = new HashSet<>();
    Set<PolymorphicComponent> removedComponents = new HashSet<>();
    int outputFanoutFactorizationRemovedCount = 0;
    int outputFanoutFactorizationAddedCount = 0;
    int outputFactorizationFanoutReduction = 0;

    for (PolymorphicComponent c : pn.getComponents())
    {
      if (c.getOutputs().size() > largeGateThreshold)
      {
        if ((c instanceof PolymorphicOr))
        {
          //	Can we find a common factor across output ANDs?
          Set<PolymorphicComponent> outputANDinputs = new HashSet<>();

          for (PolymorphicComponent output : c.getOutputs())
          {
            if (!(output instanceof PolymorphicAnd))
            {
              //	Not uniform
              continue;
            }

            if (outputANDinputs.isEmpty())
            {
              outputANDinputs.addAll(output.getInputs());
              outputANDinputs.remove(c);
            }
            else
            {
              Collection<? extends PolymorphicComponent> testInputs = output
                  .getInputs();

              for (Iterator<PolymorphicComponent> itr = outputANDinputs
                  .iterator(); itr.hasNext();)
              {
                if (!testInputs.contains(itr.next()))
                {
                  itr.remove();
                }
              }

              if (outputANDinputs.isEmpty())
              {
                break;
              }
            }
          }

          if (!outputANDinputs.isEmpty())
          {
            outputFactorizationFanoutReduction += c.getOutputs().size() - 1;

            PolymorphicAnd newAnd = pn.getComponentFactory().createAnd(-1, -1);
            newComponents.add(newAnd);
            outputFanoutFactorizationAddedCount++;

            for (PolymorphicComponent factor : outputANDinputs)
            {
              newAnd.addInput(factor);
              factor.addOutput(newAnd);
            }

            //	Remove the factors from the outputs
            Set<PolymorphicComponent> outputs = new HashSet<>(c
                .getOutputs());
            for (PolymorphicComponent output : outputs)
            {
              if (output instanceof PolymorphicAnd)
              {
                for (PolymorphicComponent factor : outputANDinputs)
                {
                  output.removeInput(factor);
                  factor.removeOutput(output);
                }

                output.removeInput(c);
                c.removeOutput(output);

                //	If it was everything just collapse it out entirely
                if (output.getInputs().isEmpty())
                {
                  removedComponents.add(output);
                  outputFanoutFactorizationRemovedCount++;

                  for (PolymorphicComponent outputOutput : output.getOutputs())
                  {
                    newAnd.addOutput(outputOutput);
                    outputOutput.addInput(newAnd);
                  }
                }
                else
                {
                  newAnd.addOutput(output);
                  output.addInput(newAnd);
                }
              }
            }

            newAnd.addInput(c);
            c.addOutput(newAnd);
          }
        }
        else if ((c instanceof PolymorphicAnd))
        {
          //	Can we find a common factor across output ORs?
          Set<PolymorphicComponent> outputORinputs = new HashSet<>();

          for (PolymorphicComponent output : c.getOutputs())
          {
            if (!(output instanceof PolymorphicOr))
            {
              //	Not uniform
              continue;
            }

            if (outputORinputs.isEmpty())
            {
              outputORinputs.addAll(output.getInputs());
              outputORinputs.remove(c);
            }
            else
            {
              Collection<? extends PolymorphicComponent> testInputs = output
                  .getInputs();

              for (Iterator<PolymorphicComponent> itr = outputORinputs
                  .iterator(); itr.hasNext();)
              {
                if (!testInputs.contains(itr.next()))
                {
                  itr.remove();
                }
              }

              if (outputORinputs.isEmpty())
              {
                break;
              }
            }
          }

          if (!outputORinputs.isEmpty())
          {
            outputFactorizationFanoutReduction += c.getOutputs().size() - 1;

            PolymorphicOr newOr = pn.getComponentFactory().createOr(-1, -1);
            newComponents.add(newOr);
            outputFanoutFactorizationAddedCount++;

            for (PolymorphicComponent factor : outputORinputs)
            {
              newOr.addInput(factor);
              factor.addOutput(newOr);
            }

            //	Remove the factors from the outputs
            Set<PolymorphicComponent> outputs = new HashSet<>(c
                .getOutputs());
            for (PolymorphicComponent output : outputs)
            {
              if (output instanceof PolymorphicOr)
              {
                for (PolymorphicComponent factor : outputORinputs)
                {
                  output.removeInput(factor);
                  factor.removeOutput(output);
                }

                output.removeInput(c);
                c.removeOutput(output);

                //	If it was everything just collapse it out entirely
                if (output.getInputs().isEmpty())
                {
                  removedComponents.add(output);
                  outputFanoutFactorizationRemovedCount++;

                  for (PolymorphicComponent outputOutput : output.getOutputs())
                  {
                    newOr.addOutput(outputOutput);
                    outputOutput.addInput(newOr);
                  }
                }
                else
                {
                  newOr.addOutput(output);
                  output.addInput(newOr);
                }
              }
            }

            newOr.addInput(c);
            c.addOutput(newOr);
          }
        }
      }
    }

    for (PolymorphicComponent c : newComponents)
    {
      pn.addComponent(c);
    }

    for (PolymorphicComponent c : removedComponents)
    {
      pn.removeComponent(c);
    }

    int outputToInputFactorizationNetRemovedCount = outputFanoutFactorizationRemovedCount -
                                                    outputFanoutFactorizationAddedCount;
    LOGGER.debug("Fanout reduction by factorization of size: " +
                 outputFactorizationFanoutReduction + " at a cost of " +
                 -outputToInputFactorizationNetRemovedCount + " gates");
  }

  /**
   * Potentially optimizes an already-existing propnet by removing propositions
   * with no special meaning. The inputs and outputs of those propositions are
   * connected to one another. This is unlikely to improve performance unless
   * values of every single component are stored (outside the propnet).
   *
   * @param pn
   */
  public static void removeAnonymousPropositions(PolymorphicPropNet pn)
  {
    List<PolymorphicProposition> toSplice = new ArrayList<>();
    List<PolymorphicProposition> toReplaceWithFalse = new ArrayList<>();
    for (PolymorphicProposition p : pn.getPropositions())
    {
      //If it's important, continue to the next proposition
      if (p.getInputs().size() == 1 &&
          p.getSingleInput() instanceof PolymorphicTransition)
        //It's a base proposition
        continue;
      GdlSentence sentence = p.getName();
      if (sentence instanceof GdlProposition)
      {
        if (sentence.getName() == GdlPool.TERMINAL || sentence.getName() == INIT_CAPS)
          continue;
      }
      else
      {
        GdlRelation relation = (GdlRelation)sentence;
        GdlConstant name = relation.getName();
        if (name == GdlPool.LEGAL || name == GdlPool.GOAL || name == GdlPool.DOES || name == GdlPool.INIT)
          continue;
      }
      if (p.getInputs().size() < 1)
      {
        //Needs to be handled separately...
        //because this is an always-false true proposition
        //and it might have and gates as outputs
        toReplaceWithFalse.add(p);
        continue;
      }
      if (p.getInputs().size() != 1)
        System.err.println("Might have falsely declared " + p.getName() +
                           " to be unimportant?");
      //Not important
      toSplice.add(p);
    }
    for (PolymorphicProposition p : toSplice)
    {
      //Get the inputs and outputs...
      Collection<? extends PolymorphicComponent> inputs = p.getInputs();
      Collection<? extends PolymorphicComponent> outputs = p.getOutputs();
      //Remove the proposition...
      pn.removeComponent(p);
      //And splice the inputs and outputs back together
      if (inputs.size() > 1)
        System.err
            .println("Programmer made a bad assumption here... might lead to trouble?");
      for (PolymorphicComponent input : inputs)
      {
        for (PolymorphicComponent output : outputs)
        {
          input.addOutput(output);
          output.addInput(input);
        }
      }
    }
    for (PolymorphicProposition p : toReplaceWithFalse)
    {
      LOGGER.warn("Should be replacing " + p +
                  " with false, but should do that in the OPNF, really; better equipped to do that there");
    }
  }

  public static void removeInitPropositions(PolymorphicPropNet propNet)
  {
    List<PolymorphicComponent> removedComponents = new LinkedList<>();

    for (PolymorphicComponent c : propNet.getComponents())
    {
      if (c instanceof PolymorphicProposition)
      {
        GdlConstant name = ((PolymorphicProposition)c).getName().getName();

        if (name == GdlPool.INIT)
        {
          if (c.getInputs().size() > 0)
          {
            c.getSingleInput().removeOutput(c);
          }
          for (PolymorphicComponent output : c.getOutputs())
          {
            output.removeInput(c);
          }

          removedComponents.add(c);
        }
      }
    }

    for (PolymorphicComponent c : removedComponents)
    {
      propNet.removeComponent(c);
    }
  }

  private static boolean hasNoNonGoalDependents(PolymorphicComponent c)
  {
    for(PolymorphicComponent output : c.getOutputs())
    {
      if ( output instanceof PolymorphicTransition )
      {
        return false;
      }
      if ( output instanceof PolymorphicProposition )
      {
        GdlConstant name = ((PolymorphicProposition)output).getName()
            .getName();

        if (name != GdlPool.GOAL)
        {
          return false;
        }
      }

      if ( !hasNoNonGoalDependents(output))
      {
        return false;
      }
    }
    return true;
  }

  public static void removeGoalPropositions(PolymorphicPropNet propNet)
  {
    for (PolymorphicProposition[] roleGoals : propNet.getGoalPropositions().values())
    {
      for (PolymorphicProposition c : roleGoals)
      {
        if ( hasNoNonGoalDependents(c) )
        {
          propNet.removeComponent(c);
        }
      }
    }
  }

  private static void removeDependencyOnLegalsAndTerminal(PolymorphicComponent c,
                                                          Set<PolymorphicComponent> inputClosure)
  {
    if (inputClosure == null)
    {
      inputClosure = new HashSet<>();
    }

    Set<PolymorphicComponent> toRewire = new HashSet<>();
    for (PolymorphicComponent input : c.getInputs())
    {
      if (!inputClosure.contains(input))
      {
        if (input instanceof PolymorphicProposition)
        {
          GdlConstant name = ((PolymorphicProposition)input).getName()
              .getName();

          if (name == GdlPool.LEGAL || name == GdlPool.TERMINAL)
          {
            toRewire.add(input);
          }
          else
          {
            inputClosure.add(input);
          }
        }
        else
        {
          inputClosure.add(input);
          if (!(input instanceof PolymorphicTransition))
          {
            removeDependencyOnLegalsAndTerminal(input, inputClosure);
          }
        }
      }
    }

    for(PolymorphicComponent input : toRewire)
    {
      c.removeInput(input);
      input.removeOutput(c);
      input.getSingleInput().addOutput(c);
      c.addInput(input.getSingleInput());
    }
  }

  /**
   * Remove things we do not need to support goal determination
   * @param propNet
   */
  public static void removeAllButGoalPropositions(PolymorphicPropNet propNet)
  {
    Set<PolymorphicComponent> removedComponents = new HashSet<>();

    for (PolymorphicComponent c : propNet.getComponents())
    {
      if (c instanceof PolymorphicProposition)
      {
        GdlConstant name = ((PolymorphicProposition)c).getName().getName();

        if (name == GdlPool.GOAL)
        {
          removeDependencyOnLegalsAndTerminal(c, null);
        }
      }
    }

    for (PolymorphicComponent c : propNet.getComponents())
    {
      boolean remove = false;

      if (c instanceof PolymorphicProposition)
      {
        GdlConstant name = ((PolymorphicProposition)c).getName().getName();

        //  We're not actually allowed to remove TERMINAL so just remove its input
        //  so, there is never any trigger cost
        if ( name == GdlPool.TERMINAL )
        {
          c.getSingleInput().removeOutput(c);
          c.removeAllInputs();
        }
        else if (name != GdlPool.TRUE &&
                 name != GdlPool.BASE &&
                 name != GdlPool.GOAL )
        {
          remove = true;
        }
      }
      else if (c instanceof PolymorphicTransition)
      {
        remove = true;
      }

      if (remove)
      {
        if (c.getInputs().size() > 0)
        {
          c.getSingleInput().removeOutput(c);
        }
        for (PolymorphicComponent output : c.getOutputs())
        {
          output.removeInput(c);
        }

        removedComponents.add(c);
      }
    }

    removeComponentsAndMinimize(propNet, removedComponents);
  }

  private static void removeComponentsAndMinimize(PolymorphicPropNet propNet, Set<PolymorphicComponent> removedComponents)
  {
    for (PolymorphicComponent c : removedComponents)
    {
      propNet.removeComponent(c);
    }

    removedComponents.clear();

    int numStartComponents;
    int numEndComponents;

    do
    {
      numStartComponents = propNet.getComponents().size();
      OptimizingPolymorphicPropNetFactory.removeRedundantConstantsAndGates(propNet, true);
      numEndComponents = propNet.getComponents().size();
    }
    while (numEndComponents != numStartComponents);

    //  Now we can trim any base props which don't feed into other logic
    for (PolymorphicComponent c : propNet.getComponents())
    {
      if (c instanceof PolymorphicProposition)
      {
        GdlConstant name = ((PolymorphicProposition)c).getName().getName();

        if ((name == GdlPool.TRUE || name == GdlPool.BASE) && c.getOutputs().isEmpty())
        {
          removedComponents.add(c);
        }
      }
    }

    for (PolymorphicComponent c : removedComponents)
    {
      propNet.removeComponent(c);
    }

    do
    {
      numStartComponents = propNet.getComponents().size();
      OptimizingPolymorphicPropNetFactory.removeRedundantConstantsAndGates(propNet, true);
      numEndComponents = propNet.getComponents().size();
    }
    while (numEndComponents != numStartComponents);
  }

  /**
   * Remove things we do not need to support terminality determination
   * Also may remove components needed to determine legals unless it appears beneficial
   * to use the reduced network in preference for that purpose
   * @param propNet
   */
  public static void removeAllButTerminalProposition(PolymorphicPropNet propNet)
  {
    Set<PolymorphicComponent> removedComponents = new HashSet<>();

    for (PolymorphicComponent c : propNet.getComponents())
    {
      boolean remove = false;

      if (c instanceof PolymorphicProposition)
      {
        GdlConstant name = ((PolymorphicProposition)c).getName().getName();

        if (name != GdlPool.TRUE &&
            name != GdlPool.BASE &&
            name != GdlPool.TERMINAL)
        {
          remove = true;
        }
      }
      else if (c instanceof PolymorphicTransition)
      {
        remove = true;
      }

      if (remove)
      {
        if (c.getInputs().size() > 0)
        {
          c.getSingleInput().removeOutput(c);
        }
        for (PolymorphicComponent output : c.getOutputs())
        {
          output.removeInput(c);
        }

        removedComponents.add(c);
      }
    }

    removeComponentsAndMinimize(propNet, removedComponents);
  }

  public static void fixBaseProposition(PolymorphicPropNet propNet,
                                        GdlSentence propName,
                                        boolean value)
  {
    LOGGER.debug("Hardwire base prop " + propName + " to value: " + value);
    PolymorphicProposition prop = propNet.getBasePropositions().get(propName);
    PolymorphicConstant replacement = propNet.getComponentFactory()
        .createConstant(-1, value);

    propNet.addComponent(replacement);

    for (PolymorphicComponent c : prop.getOutputs())
    {
      c.removeInput(prop);
      c.addInput(replacement);
      replacement.addOutput(c);
    }

    prop.removeAllOutputs();

    minimizeNetwork(propNet);
  }

  public static void minimizeNetwork(PolymorphicPropNet propNet)
  {
    int numStartComponents;
    int numEndComponents;

    do
    {
      numStartComponents = propNet.getComponents().size();
      OptimizingPolymorphicPropNetFactory.removeUnreachableBasesAndInputs(propNet, true);
      OptimizingPolymorphicPropNetFactory.removeRedundantConstantsAndGates(propNet, true);
      numEndComponents = propNet.getComponents().size();
    }
    while (numEndComponents != numStartComponents);

    assert(propNet.validateClosure());
  }

  private static boolean loopsFound;

  public static void removeDuplicateLogic(PolymorphicPropNet pn)
  {
    Map<Long, List<PolymorphicComponent>> componentSignatureMap = new HashMap<>();

    loopsFound = false;
    for (PolymorphicComponent c : pn.getComponents())
    {
      calculateSignature(c, componentSignatureMap);
    }

    int duplicateCount = 0;

    for (Entry<Long, List<PolymorphicComponent>> e : componentSignatureMap
        .entrySet())
    {
      if (e.getValue().size() > 1)
      {
        //	Replace each instance with the first
        PolymorphicComponent keeper = null;

        for (PolymorphicComponent c : e.getValue())
        {
          if (keeper == null)
          {
            keeper = c;
          }
          else
          {
            duplicateCount++;

            for (PolymorphicComponent output : c.getOutputs())
            {
              output.removeInput(c);
              if (!output.getInputs().contains(keeper))
              {
                output.addInput(keeper);
                keeper.addOutput(output);
              }
            }

            pn.removeComponent(c);
          }
        }
      }
    }

    LOGGER.debug("Removed " + duplicateCount + " duplicate components");
  }

  private static Map<Class<?>, Long> componentTypeBaseSignatures = new HashMap<>();

  private static class FastHasher
  {
    private final long m       = ((long)0x880355f2) << 32 + 0x1e6d1965;
    private final long mixMult = ((long)0x2127599b) << 32 + 0xf4325c37;
    private long       h       = m;

    public FastHasher()
    {
    }

    private long mix(long x)
    {
      x ^= (x >> 23);
      x *= mixMult;
      x ^= (x >> 47);

      return x;
    }

    public void addLong(long value)
    {
      h ^= mix(value);
      h *= m;
    }

    public long getValue()
    {
      return h;
    }
  }

  private static Random rand = new Random();

  private static long calculateSignature(PolymorphicComponent c,
                                         Map<Long, List<PolymorphicComponent>> componentSignatureMap)
  {
    if (c.getSignature() == 0)
    {
      c.setSignature(2);

      FastHasher chks = new FastHasher();
      Long componentTypeSignature = componentTypeBaseSignatures.get(c
          .getClass());

      if (componentTypeSignature == null)
      {
        componentTypeSignature = new Long(rand.nextLong());
        componentTypeBaseSignatures.put(c.getClass(), componentTypeSignature);
      }

      chks.addLong(componentTypeSignature);

      long inputsSig = 0;
      int numNonTransitionalInputs = 0;

      for (PolymorphicComponent input : c.getInputs())
      {
        if (!(input instanceof PolymorphicTransition))
        {
          long inputSig = calculateSignature(input, componentSignatureMap);

          if ( inputSig == 2 )
          {
            if ( !loopsFound )
            {
              loopsFound = true;
              LOGGER.warn("Propnet loops detected - unable to check for duplicate logic amongst components fed from such loops");
            }
            return 2;
          }
          numNonTransitionalInputs++;
          inputsSig += inputSig;
        }
      }

      if (numNonTransitionalInputs == 0)
      {
        //	Effectively a logic root - needs a unique id
        inputsSig = rand.nextLong();
      }

      chks.addLong(inputsSig);

      //	Always set lowest order bit to ensure 0 cannot result
      c.setSignature(chks.getValue() | 1);

      //	Propositions are never removed for representing the same logical value as another
      //	component, nor are they used as a source
      if (!(c instanceof PolymorphicProposition) &&
          !(c instanceof PolymorphicTransition))
      {
        List<PolymorphicComponent> sigMatchList = componentSignatureMap.get(c
            .getSignature());
        if (sigMatchList == null)
        {
          sigMatchList = new LinkedList<>();
          componentSignatureMap.put(c.getSignature(), sigMatchList);
        }
        else
        {
          //	Sanity check match with first instance
          PolymorphicComponent first = sigMatchList.get(0);

          if (first.getClass() != c.getClass())
          {
            LOGGER.warn("Signature mismatch (class)");
          }
          if (first.getInputs().size() != c.getInputs().size())
          {
            LOGGER.warn("Signature mismatch (input arity)");
          }
          for (PolymorphicComponent input : c.getInputs())
          {
            boolean found = false;
            for (PolymorphicComponent other : first.getInputs())
            {
              if (other.getSignature() == input.getSignature())
              {
                found = true;
                break;
              }
            }

            if (!found)
            {
              LOGGER.warn("Signature mismatch (input mismatch)");

              long otherChk = 0;
              for (PolymorphicComponent other : first.getInputs())
              {
                LOGGER.warn("...first instance input sig: " + Long.toHexString(other.getSignature()));
                otherChk += other.getSignature();
              }
              long thisChk = 0;
              for (PolymorphicComponent cInput : c.getInputs())
              {
                LOGGER.warn("...this instance input sig: " + Long.toHexString(cInput.getSignature()));
                thisChk += cInput.getSignature();
              }
              LOGGER.warn("Aggregate inputs sigs: " +
                          Long.toHexString(otherChk) + " and " +
                          Long.toHexString(thisChk));

              FastHasher validateChk = new FastHasher();

              validateChk.addLong(componentTypeSignature);
              validateChk.addLong(otherChk);
              if (first.getSignature() != (validateChk.getValue() | 1))
              {
                LOGGER.warn("First instance's signature does not match");
              }

              validateChk = new FastHasher();

              validateChk.addLong(componentTypeSignature);
              validateChk.addLong(thisChk);
              if (c.getSignature() != (validateChk.getValue() | 1))
              {
                LOGGER.warn("This instance's signature does not match");
              }
              LOGGER.error("All fucked up");
              break;
            }
          }
        }

        sigMatchList.add(c);
      }
    }

    return c.getSignature();
  }

  public static void removeNonBaseOrDoesPropositionOutputs(PolymorphicPropNet pn)
  {
    for(PolymorphicComponent c : pn.getComponents())
    {
      if ( c instanceof PolymorphicProposition &&
           !pn.getBasePropositions().containsValue(c) &&
           !pn.getInputPropositions().containsValue(c) &&
           pn.getInitProposition() != c )
      {
        PolymorphicComponent input = c.getSingleInput();

        for(PolymorphicComponent output : c.getOutputs())
        {
          output.removeInput(c);
          output.addInput(input);
          input.addOutput(output);
        }

        c.removeAllOutputs();
      }
    }
  }

  public static void optimizeInputSets(PolymorphicPropNet pn)
  {
    //	First find the input proposition sets for each role
    Map<Role, Set<PolymorphicProposition>> roleInputMap = new HashMap<>();

    for (Role role : pn.getRoles())
    {
      roleInputMap.put(role, new HashSet<PolymorphicProposition>());
    }

    for (Entry<GdlSentence, PolymorphicProposition> e : pn
        .getInputPropositions().entrySet())
    {
      Role role = null;

      for (Role r : pn.getRoles())
      {
        if (GdlUtils.containsTerm(e.getKey(), r.getName()))
        {
          role = r;
          break;
        }
      }

      if (role == null)
      {
        LOGGER.warn("Unexpectedly could not find role of input prop");
      }

      roleInputMap.get(role).add(e.getValue());
    }

    int minInputSizeToConsider = Integer.MAX_VALUE;
    for (Set<PolymorphicProposition> roleSet : roleInputMap.values())
    {
      if (roleSet.size() < minInputSizeToConsider)
      {
        minInputSizeToConsider = roleSet.size() / 2;
      }
    }

    //	Now we know that at least one input prop must be activated by each role each turn
    //	so search for large conjuncts or disjuncts of input props and consider replacing
    //	them by their complements
//    int possibleSavings = 0;
//
//    for (PolymorphicComponent c : pn.getComponents())
//    {
//      if ((c instanceof PolymorphicAnd || c instanceof PolymorphicOr) &&
//          c.getInputs().size() > minInputSizeToConsider)
//      {
//        for (Set<PolymorphicProposition> roleSet : roleInputMap.values())
//        {
//          Set<PolymorphicComponent> inputSet = new HashSet<PolymorphicComponent>();
//
//          for (PolymorphicComponent input : c.getInputs())
//          {
//            if (input instanceof PolymorphicProposition &&
//                roleSet.contains(input))
//            {
//              inputSet.add(input);
//            }
//          }
//
//          if (inputSet.size() > roleSet.size() / 2)
//          {
//            possibleSavings += 2 * inputSet.size() - roleSet.size();
//          }
//        }
//      }
//    }

    Set<PolymorphicComponent> allInputs = new HashSet<>();

    //  Noops can present some issues for factorization in terms of replacement
    //  or ORs of does props by the not of the OR of their converse.  This is because
    //  the ability to artificially noop in one factor (for games where there is choice
    //  of which factor to play in) means that it is no longer true that SOME move is played
    //  by each player in every factor.  We treat this somewhat empirically by special-casing
    //  noops and coping with the no-legal-moves case in state transition.  To do this we
    //  must not add noop to the converse sets when replacing large ORs of does props except
    //  in the very special case where it was originally an OR of everything except noop.
    //  Accordingly we need to identify the noops - for now we use the rather iffy test of
    //  their being a does prop with arity 0 in the move specifier.
    //  TODO - replace this by a somewhat more robust dependency-based approach (a does prop on
    //  which no base props are dependent can be considered a noop)
    Set<PolymorphicComponent> noops = new HashSet<>();
    for (Set<PolymorphicProposition> roleInputs : roleInputMap.values())
    {
      allInputs.addAll(roleInputs);

      for(PolymorphicProposition input : roleInputs)
      {
        if ( input.getName().get(1).toSentence().arity() == 0 )
        {
          noops.add(input);
        }
      }
    }

    Set<PolymorphicComponent> newComponents = new HashSet<>();
    Set<PolymorphicComponent> removedComponents = new HashSet<>();

    for(PolymorphicComponent c : pn.getComponents())
    {
      if ( c instanceof PolymorphicOr )
      {
        Map<Role,Set<PolymorphicComponent>> disjunctiveInputs = new HashMap<>();

        for (Role role : pn.getRoles())
        {
          Set<PolymorphicProposition> roleInputs = roleInputMap.get(role);
          Set<PolymorphicComponent> inputSet = new HashSet<>();

          if ( !recursiveCalculateDisjunctiveRoleInputs(c, inputSet, roleInputs, allInputs) )
          {
            disjunctiveInputs.clear();
            break;
          }

          if ( inputSet.size() > (3*roleInputs.size())/4 )
          {
            disjunctiveInputs.put(role, inputSet);
          }
        }

        if ( !disjunctiveInputs.isEmpty())
        {
          Set<PolymorphicComponent> newNots = new HashSet<>();

          removedComponents.add(c);

          //  If we wind up with JUST noops for all roles then actually
          //  since all roles must have a move this is just TRUE, so detect this
          //  case.  Also note that if we DO perform this optimization then the
          //  net is no longer suitable for factorization because the introduction
          //  of a factor pseudo-noop will break the all-must-have-a-move assumption
          //  TODO - this case is really rare (just Pentago/PentagoSuicide that I know of)
          //  and does not occur in any known factorized game, but at some point we should add
          //  a factorization inhibitor such that if a game arises in which the network does
          //  go through this transform then it cannot be played factored
          boolean noopsOnly = true;
          for (Role role : pn.getRoles())
          {
            Set<PolymorphicComponent> roleInputs = disjunctiveInputs.get(role);

            if ( roleInputs != null )
            {
              PolymorphicOr newOr = pn.getComponentFactory().createOr(-1, -1);

              for(PolymorphicComponent input : roleInputMap.get(role))
              {
                if ( !roleInputs.contains(input) )
                {
                  //  Don't add noops to the converse sets - this is somewhat
                  //  empirical and it would be possible to construct GDL it doesn't work for
                  //  but it works for the typical idioms that result from distincts and
                  //  tend to give rise to these large ORs
                  if ( noops.contains(input) )
                  {
                    continue;
                  }
                  newOr.addInput(input);
                  input.addOutput(newOr);
                }
                else
                {
                  c.removeInput(input);
                  input.removeOutput(c);
                }
              }

              //  In the special case where nothing is left we need to use the noop since
              //  it was an OR of every other move previously
              if ( newOr.getInputs().isEmpty())
              {
                //  Asymmetric cases where it was all moves for one role but a specific
                //  subset for another should NOT include the noop for the non-gating role
                if ( noopsOnly )
                {
                  for(PolymorphicComponent noop : noops)
                  {
                    if ( roleInputMap.get(role).contains(noop) && !roleInputs.contains(noop))
                    {
                      newOr.addInput(noop);
                      noop.addOutput(newOr);
                    }
                  }
                }
              }
              else
              {
                //  Asymmetric cases where it was all moves for one role but a specific
                //  subset for another should NOT include the noop for the non-gating role
                if ( noopsOnly && !newNots.isEmpty() )
                {
                  newNots.clear();
                }
                noopsOnly = false;

                if ( disjunctiveInputs.size() == 1 )
                {
                  for(PolymorphicComponent noop : noops)
                  {
                    if ( roleInputMap.get(role).contains(noop) && !roleInputs.contains(noop))
                    {
                      newOr.addInput(noop);
                      noop.addOutput(newOr);
                    }
                  }
                }
              }

              if ( !newOr.getInputs().isEmpty() )
              {
                PolymorphicNot newNot = pn.getComponentFactory().createNot(-1);

                if ( newOr.getInputs().size() > 1 )
                {
                  newComponents.add(newOr);
                  newOr.addOutput(newNot);
                  newNot.addInput(newOr);
                }
                else
                {
                  PolymorphicComponent source = newOr.getSingleInput();

                  source.removeOutput(newOr);
                  source.addOutput(newNot);
                  newNot.addInput(source);
                }

                newNots.add(newNot);
                newComponents.add(newNot);
              }
            }
          }

          PolymorphicComponent newOutput;

          if ( newNots.size() == 1 )
          {
            newOutput = newNots.iterator().next();
          }
          else
          {
            if ( noopsOnly )
            {
              newOutput = pn.getComponentFactory().createConstant(-1, true);
            }
            else
            {
              newOutput = pn.getComponentFactory().createAnd(-1, -1);

              for(PolymorphicComponent newNot : newNots)
              {
                newNot.addOutput(newOutput);
                newOutput.addInput(newNot);
              }
            }
            newComponents.add(newOutput);
          }

          for(PolymorphicComponent output : c.getOutputs())
          {
            newOutput.addOutput(output);
            output.removeInput(c);
            output.addInput(newOutput);
          }

          c.removeAllOutputs();
        }
      }
    }

    for(PolymorphicComponent c : newComponents)
    {
      pn.addComponent(c);
    }

    for(PolymorphicComponent c : removedComponents)
    {
      pn.removeComponent(c);
    }
  }

  private static boolean recursiveCalculateDisjunctiveRoleInputs(PolymorphicComponent c, Set<PolymorphicComponent> inputs, Set<PolymorphicProposition> inputSet, Set<PolymorphicComponent> allowableInputs)
  {
    if ( c instanceof PolymorphicOr )
    {
      for(PolymorphicComponent input : c.getInputs())
      {
        if ( !recursiveCalculateDisjunctiveRoleInputs(input, inputs, inputSet, allowableInputs))
        {
          return false;
        }
      }

      return true;
    }
    else if ( c instanceof PolymorphicProposition )
    {
      if ( !allowableInputs.contains(c) )
      {
        return false;
      }

      if ( inputSet.contains(c))
      {
        inputs.add(c);
      }

      return true;
    }

    return false;
  }

  public static void optimizeInvertedInputs(PolymorphicPropNet pn)
  {
    Set<PolymorphicComponent> refactoredComponents = new HashSet<>();
    Set<PolymorphicComponent> newComponents = new HashSet<>();

    for (PolymorphicComponent c : pn.getComponents())
    {
      int numInvertedInputs = 0;

      //  Special case - don't refactor ORs leading directly into the
      //  terminal prop or we'll screw up factor analysis!
      if ( c instanceof PolymorphicOr && c.getOutputs().contains(pn.getTerminalProposition()))
      {
        continue;
      }

      for (PolymorphicComponent input : c.getInputs())
      {
        if (input instanceof PolymorphicNot)
        {
          numInvertedInputs++;
        }
      }

      if (numInvertedInputs > (c.getInputs().size() * 3) / 4)
      {
        //	For now just do the trivial case
        if (numInvertedInputs == c.getInputs().size())
        {
          if (numInvertedInputs > 1)
          {
            refactoredComponents.add(c);
          }
        }
      }
    }

    for (PolymorphicComponent c : refactoredComponents)
    {
      if (c instanceof PolymorphicAnd)
      {
        PolymorphicOr newOr = pn.getComponentFactory().createOr(-1, -1);
        PolymorphicNot newNot = pn.getComponentFactory().createNot(-1);

        newComponents.add(newOr);
        newComponents.add(newNot);

        for (PolymorphicComponent output : c.getOutputs())
        {
          output.addInput(newNot);
          newNot.addOutput(output);
        }

        newOr.addOutput(newNot);
        newNot.addInput(newOr);

        Set<PolymorphicComponent> removedNots = new HashSet<>();

        for (PolymorphicComponent inputNot : c.getInputs())
        {
          if (!(inputNot instanceof PolymorphicNot))
          {
            throw new RuntimeException("Unhandled case of inverted input removal - not a NOT!");
          }

          newOr.addInput(inputNot.getSingleInput());
          inputNot.getSingleInput().addOutput(newOr);

          if (inputNot.getOutputs().size() == 1)
          {
            removedNots.add(inputNot);
          }
        }

        for (PolymorphicComponent inputNot : removedNots)
        {
          pn.removeComponent(inputNot);
        }

        pn.removeComponent(c);
      }
      else if (c instanceof PolymorphicOr)
      {
        PolymorphicAnd newAnd = pn.getComponentFactory().createAnd(-1, -1);
        PolymorphicNot newNot = pn.getComponentFactory().createNot(-1);

        newComponents.add(newAnd);
        newComponents.add(newNot);

        for (PolymorphicComponent output : c.getOutputs())
        {
          output.addInput(newNot);
          newNot.addOutput(output);
        }

        newAnd.addOutput(newNot);
        newNot.addInput(newAnd);

        Set<PolymorphicComponent> removedNots = new HashSet<>();

        for (PolymorphicComponent inputNot : c.getInputs())
        {
          if (!(inputNot instanceof PolymorphicNot))
          {
            throw new RuntimeException("Unhandled case of inverted input removal - not a NOT!");
          }

          newAnd.addInput(inputNot.getSingleInput());
          inputNot.getSingleInput().addOutput(newAnd);

          if (inputNot.getOutputs().size() == 1)
          {
            removedNots.add(inputNot);
          }
        }

        for (PolymorphicComponent inputNot : removedNots)
        {
          pn.removeComponent(inputNot);
        }

        pn.removeComponent(c);
      }
    }

    for (PolymorphicComponent c : newComponents)
    {
      pn.addComponent(c);
    }
  }
}
