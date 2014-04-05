package org.ggp.base.player.gamer.statemachine.sancho;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.ggp.base.player.gamer.statemachine.sancho.TreeNode.TreeNodeRef;
import org.ggp.base.player.gamer.statemachine.sancho.heuristic.Heuristic;
import org.ggp.base.util.profile.ProfileSection;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.Factor;
import org.ggp.base.util.statemachine.implementation.propnet.forwardDeadReckon.ForwardDeadReckonPropnetStateMachine;

public class MCTSTree
{
  class LRUNodeMoveWeightsCache
  extends
  LinkedHashMap<TreeNode, MoveWeightsCollection>
  {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private int               maxEntries;

    public LRUNodeMoveWeightsCache(int capacity)
    {
      super(capacity + 1, 1.0f, true);
      maxEntries = capacity;
    }

    @Override
    protected boolean removeEldestEntry(final Map.Entry<TreeNode, MoveWeightsCollection> eldest)
    {
      return super.size() > maxEntries;
    }
  }

  class MoveScoreInfo
  {
    public double averageScore = 0;
    public double sampleWeight = 0;
  }


  final boolean                                        freeCompletedNodeChildren                   = true;                                                          //true;
  final boolean                                        disableOnelevelMinimax                      = true;  //false;
  ForwardDeadReckonPropnetStateMachine                 underlyingStateMachine;
  volatile TreeNode                                    root = null;
  int                                                  numRoles;
  LRUNodeMoveWeightsCache                              nodeMoveWeightsCache                        = null;
  NodePool                                             nodePool;
  Map<ForwardDeadReckonInternalMachineState, TreeNode> positions                                   = new HashMap<>();
  int                                                  sweepInstance                               = 0;
  List<TreeNode>                                       completedNodeQueue                          = new LinkedList<>();
  Map<Move, MoveScoreInfo>                             cousinMoveCache                             = new HashMap<>();
  TreeNodeRef                                          cousinMovesCachedFor                        = null;
  double[]                                             bonusBuffer                                 = null;
  double[]                                             roleRationality                             = null;
  long                                                 numCompletionsProcessed                     = 0;
  Random                                               r                                           = new Random();
  int                                                  numUniqueTreeNodes                          = 0;
  int                                                  numTotalTreeNodes                           = 0;
  int                                                  numTerminalRollouts                         = 0;
  int                                                  numNonTerminalRollouts                      = 0;
  int                                                  numIncompleteNodes                          = 0;
  int                                                  numCompletedBranches                        = 0;
  boolean                                              completeSelectionFromIncompleteParentWarned = false;
  int                                                  numSelectionsThroughIncompleteNodes         = 0;
  int                                                  numReExpansions                             = 0;
  Heuristic                                            heuristic;
  RoleOrdering                                         roleOrdering;
  RolloutProcessorPool                                 rolloutPool;
  RuntimeGameCharacteristics                           gameCharacteristics;
  Factor                                               factor;

  public MCTSTree(ForwardDeadReckonPropnetStateMachine stateMachine,
                  Factor factor,
                  NodePool nodePool,
                  RoleOrdering roleOrdering,
                  RolloutProcessorPool rolloutPool,
                  RuntimeGameCharacteristics gameCharacateristics,
                  Heuristic heuristic)
  {
    underlyingStateMachine = stateMachine;
    numRoles = stateMachine.getRoles().size();
    this.nodePool = nodePool;
    this.factor = factor;
    this.roleOrdering = roleOrdering;
    this.heuristic = heuristic;
    this.gameCharacteristics = gameCharacateristics;
    this.rolloutPool = rolloutPool;

    nodeMoveWeightsCache = new LRUNodeMoveWeightsCache(5000);

    bonusBuffer = new double[numRoles];
    roleRationality = new double[numRoles];
    numCompletionsProcessed = 0;
    completeSelectionFromIncompleteParentWarned = false;

    //  For now assume players in muli-player games are somewhat irrational.
    //  FUTURE - adjust during the game based on correlations with expected
    //  scores
    for (int i = 0; i < numRoles; i++)
    {
      if (gameCharacateristics.isMultiPlayer)
      {
        roleRationality[i] = (i == 0 ? 1 : 0.8);
      }
      else
      {
        roleRationality[i] = 1;
      }
    }
  }

  public void empty()
  {
    numUniqueTreeNodes = 0;
    numTotalTreeNodes = 0;
    numCompletedBranches = 0;
    root = null;
    nodePool.clear(this);
    positions.clear();
    numIncompleteNodes = 0;
    if (nodeMoveWeightsCache != null)
    {
      nodeMoveWeightsCache.clear();
    }
  }

  TreeNode allocateNode(ForwardDeadReckonPropnetStateMachine underlyingStateMachine,
                                ForwardDeadReckonInternalMachineState state,
                                TreeNode parent)
      throws GoalDefinitionException
  {
    ProfileSection methodSection = new ProfileSection("allocateNode");
    try
    {
      TreeNode result = (state != null ? positions.get(state) : null);

      //validateAll();
      numTotalTreeNodes++;
      if (result == null)
      {
        numUniqueTreeNodes++;

        //System.out.println("Add state " + state);
        result = nodePool.allocateNode(this);
        result.state = state;

        //if ( positions.values().contains(result))
        //{
        //  System.out.println("Node already referenced by a state!");
        //}
        if (state != null)
        {
          positions.put(state, result);
        }
      }
      else
      {
        if (result.freed)
        {
          System.out.println("Bad ref in positions table!");
        }
        if (result.decidingRoleIndex != numRoles - 1)
        {
          System.out.println("Non-null move in position cache");
        }
      }

      if (parent != null)
      {
        result.parents.add(parent);

        //parent.adjustDescendantCounts(result.descendantCount+1);
      }

      //validateAll();
      return result;
    }
    finally
    {
      methodSection.exitScope();
    }
  }

  void processNodeCompletions()
  {
    while (!completedNodeQueue.isEmpty())
    {
      //validateAll();
      TreeNode node = completedNodeQueue.remove(0);

      if (!node.freed)
      {
        node.processCompletion();
      }
    }
  }

  public void setRootState(ForwardDeadReckonInternalMachineState state) throws GoalDefinitionException
  {
    ForwardDeadReckonInternalMachineState factorState;

    if ( factor == null )
    {
      factorState = state;
    }
    else
    {
      factorState = new ForwardDeadReckonInternalMachineState(state);
      factorState.intersect(factor.getStateMask());
    }

    if (root == null)
    {
      root = allocateNode(underlyingStateMachine, factorState, null);
      root.decidingRoleIndex = numRoles - 1;
    }
    else
    {
      TreeNode newRoot = root.findNode(factorState,
                                       underlyingStateMachine.getRoles()
                                           .size() + 1);
      if (newRoot == null)
      {
        System.out.println("Unable to find root node in existing tree");
        empty();
        root = allocateNode(underlyingStateMachine, factorState, null);
        root.decidingRoleIndex = numRoles - 1;
      }
      else
      {
        if (newRoot != root)
        {
          root.freeAllBut(newRoot);

          root = newRoot;
        }
      }
    }
    //validateAll();

    if (root.complete && root.children == null)
    {
      System.out
          .println("Encountered complete root with trimmed children - must re-expand");
      root.complete = false;
      numCompletedBranches--;
    }

    heuristic.newTurn(root.state, root);
  }

  public boolean growTree()
      throws MoveDefinitionException, TransitionDefinitionException,
      GoalDefinitionException, InterruptedException
  {
    //validateAll();
    //validationCount++;
    if (!rolloutPool.isBackedUp())
    {
      root.selectAction();

      processNodeCompletions();
    }

    return root.complete;
  }

  Move getBestMove()
  {
    Move bestMove = root.getBestMove(true, null);

    System.out.println("Num total tree node allocations: " +
        numTotalTreeNodes);
    System.out.println("Num unique tree node allocations: " +
        numUniqueTreeNodes);
    System.out.println("Num true rollouts added: " + numNonTerminalRollouts);
    System.out.println("Num terminal nodes revisited: " +
        numTerminalRollouts);
    System.out.println("Num incomplete nodes: " + numIncompleteNodes);
    System.out.println("Num selections through incomplete nodes: " +
        numSelectionsThroughIncompleteNodes);
    System.out.println("Num node re-expansions: " + numReExpansions);
    System.out.println("Num completely explored branches: " +
        numCompletedBranches);
    System.out
    .println("Current rollout sample size: " + gameCharacteristics.getRolloutSampleSize());
    System.out.println("Current observed rollout score range: [" +
        rolloutPool.lowestRolloutScoreSeen + ", " +
        rolloutPool.highestRolloutScoreSeen + "]");
    System.out.println("Heuristic bias: " + heuristic.getSampleWeight());

    numSelectionsThroughIncompleteNodes = 0;
    numReExpansions = 0;
    numNonTerminalRollouts = 0;
    numTerminalRollouts = 0;
    return bestMove;
  }

  void validateAll()
  {
    if (root != null)
      root.validate(true);

    for (Entry<ForwardDeadReckonInternalMachineState, TreeNode> e : positions
        .entrySet())
    {
      if (e.getValue().decidingRoleIndex != numRoles - 1)
      {
        System.out.println("Position references bad type");
      }
      if (!e.getValue().state.equals(e.getKey()))
      {
        System.out.println("Position state mismatch");
      }
    }

    int incompleteCount = 0;

    for (TreeNode node : nodePool.getNodesTable())
    {
      if (node != null && !node.freed)
      {
        if (node.trimmedChildren > 0 && !node.complete)
        {
          incompleteCount++;
        }
        if (node.decidingRoleIndex == numRoles - 1)
        {
          if (node != positions.get(node.state))
          {
            System.out.println("Missing reference in positions table");
            System.out.print("node state is: " + node.state + " with hash " +
                             node.state.hashCode());
            System.out.print(positions.get(node.state));
          }
        }
      }
    }

    if (incompleteCount != numIncompleteNodes)
    {
      System.out.println("Incomplete count mismatch");
    }
  }
}
