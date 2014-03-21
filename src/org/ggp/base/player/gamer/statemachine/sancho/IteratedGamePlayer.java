package org.ggp.base.player.gamer.statemachine.sancho;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.ggp.base.player.gamer.statemachine.StateMachineGamer;
import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.statemachine.MachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;
import org.ggp.base.util.statemachine.implementation.propnet.TestForwardDeadReckonPropnetStateMachine;

public class IteratedGamePlayer
{
  private TestForwardDeadReckonPropnetStateMachine underlyingStateMachine;
  private StateMachineGamer gamer;
  private boolean isPseudoSimultaneousMove;
  private RoleOrdering roleOrdering;
  private Role ourRole;
  private double competitivenessBonus;
  private Random r = new Random();

  public IteratedGamePlayer(TestForwardDeadReckonPropnetStateMachine stateMachine,
                            StateMachineGamer gamer,
                            boolean isPseudoSimultaneousMove,
                            RoleOrdering roleOrdering,
                            double competitivenessBonus)
  {
    underlyingStateMachine = stateMachine;
    this.gamer = gamer;
    this.isPseudoSimultaneousMove = isPseudoSimultaneousMove;
    this.roleOrdering  = roleOrdering;
    this.competitivenessBonus = competitivenessBonus;

    ourRole = roleOrdering.roleIndexToRole(0);
  }

  public Move selectMove(List<Move> moves, long timeout)
      throws MoveDefinitionException, TransitionDefinitionException,
      GoalDefinitionException
  {
    List<List<GdlTerm>> moveHistory = gamer.getMatch().getMoveHistory();
    List<Set<GdlSentence>> stateHistory = gamer.getMatch().getStateHistory();
    Move bestMove = null;
    Map<Move, Map<Move, Integer>> opponentMoveSelectionCounts = new HashMap<>();
    Move lastPlayedOpponentChoice = null;
    Move lastPlayedOurChoice = null;
    int lastPlayedOurChoiceSize = 0;

    if (moves.size() == 1)
    {
      return moves.get(0);
    }

    boolean responderInNonSimultaneousGame = (!isPseudoSimultaneousMove && moveHistory
        .size() % 2 == 1);

    for (int i = 0; i < moveHistory.size(); i++)
    {
      List<GdlTerm> moveTerms = moveHistory.get(i);
      MachineState state = new MachineState(stateHistory.get(i));
      List<Move> historicalJointMove = new ArrayList<>();
      for (GdlTerm sentence : moveTerms)
      {
        historicalJointMove.add(underlyingStateMachine
            .getMoveFromTerm(sentence));
      }

      //  Did our opponent have a choice?
      if (underlyingStateMachine.getLegalMoves(state, roleOrdering.roleIndexToRole(1))
          .size() > 1)
      {
        lastPlayedOpponentChoice = historicalJointMove
            .get(roleOrdering.roleIndexToRawRoleIndex(1));

        //  Was this following a previous move of ours?  If so bump the response counts
        //  for that move
        if (lastPlayedOurChoice != null)
        {
          Map<Move, Integer> moveWeights = opponentMoveSelectionCounts
              .get(lastPlayedOurChoice);
          if (moveWeights == null)
          {
            moveWeights = new HashMap<>();
            opponentMoveSelectionCounts.put(lastPlayedOurChoice, moveWeights);
          }

          Integer val = moveWeights.get(lastPlayedOpponentChoice);

          moveWeights
              .put(lastPlayedOpponentChoice,
                   (val == null ? lastPlayedOurChoiceSize : val +
                                                            lastPlayedOurChoiceSize));
        }
      }

      //  Did we have a choice?
      if (underlyingStateMachine.getLegalMoves(state, roleOrdering.roleIndexToRole(0))
          .size() > 1)
      {
        lastPlayedOurChoice = historicalJointMove
            .get(roleOrdering.roleIndexToRawRoleIndex(0));
        lastPlayedOurChoiceSize = underlyingStateMachine
            .getLegalMoves(state, roleOrdering.roleIndexToRole(0)).size();

        if (lastPlayedOpponentChoice != null)
        {
          //  Bump for all moves we can play the count for the move the opponent played last
          for (Move legalMove : underlyingStateMachine
              .getLegalMoves(state, roleOrdering.roleIndexToRole(0)))
          {
            Map<Move, Integer> moveWeights = opponentMoveSelectionCounts
                .get(legalMove);
            if (moveWeights == null)
            {
              moveWeights = new HashMap<>();
              opponentMoveSelectionCounts.put(legalMove, moveWeights);
            }

            Integer val = moveWeights.get(lastPlayedOpponentChoice);

            moveWeights.put(lastPlayedOpponentChoice, (val == null ? 1
                                                                  : val + 1));
          }
        }
      }
    }

    ForwardDeadReckonInternalMachineState currentState = underlyingStateMachine
        .createInternalState(gamer.getCurrentState());

    int bestScore = -1;
    int opponentScoreAtBestScore = 0;
    for (Move move : moves)
    {
      ForwardDeadReckonInternalMachineState state = new ForwardDeadReckonInternalMachineState(currentState);
      Map<Move, Integer> moveWeights = opponentMoveSelectionCounts.get(move);

      System.out.println("Considering move: " + move);
      if (responderInNonSimultaneousGame)
      {
        System.out
            .println("We are responder so assuming opponent continues to play " +
                     lastPlayedOpponentChoice);
      }
      else if (moveWeights != null)
      {
        System.out.println("Response weights: " + moveWeights.values());
      }

      while (!underlyingStateMachine.isTerminal(state))
      {
        Move[] jointMove = new Move[2];
        int roleIndex = 0;

        for (Role role : underlyingStateMachine.getRoles())
        {
          List<Move> roleMoves = underlyingStateMachine.getLegalMoves(state,
                                                                      role);

          if (roleMoves.size() == 1)
          {
            jointMove[roleIndex] = roleMoves.get(0);
          }
          else if (role.equals(ourRole))
          {
            if (!roleMoves.contains(move))
            {
              System.out
                  .println("Unexpectedly cannot play intended move in iterated game!");
            }
            jointMove[roleIndex] = move;
          }
          else if (responderInNonSimultaneousGame)
          {
            jointMove[roleIndex] = lastPlayedOpponentChoice;
          }
          else
          {
            //  Do we have opponent response stats for this move?
            if (moveWeights == null)
            {
              //  Assume flat distribution
              moveWeights = new HashMap<>();

              for (Move m : roleMoves)
              {
                moveWeights.put(m, 1);
              }
            }

            int total = 0;
            for (Integer weight : moveWeights.values())
            {
              total += weight;
            }

            int rand = r.nextInt(total);
            for (Move m : roleMoves)
            {
              Integer weight = moveWeights.get(m);
              if (weight != null)
              {
                rand -= weight;
              }

              if (rand < 0)
              {
                jointMove[roleIndex] = m;
                break;
              }
            }
          }

          roleIndex++;
        }

        state = underlyingStateMachine.getNextState(state, jointMove);
      }

      int score = underlyingStateMachine.getGoal(ourRole);
      int opponentScore = underlyingStateMachine.getGoal(roleOrdering.roleIndexToRole(1));

      if (score >= opponentScore)
      {
        score += competitivenessBonus;
        if (score > opponentScore)
        {
          score += competitivenessBonus;
        }
      }
      if (score > bestScore ||
          (score == bestScore && opponentScoreAtBestScore > opponentScore))
      {
        bestScore = score;
        opponentScoreAtBestScore = opponentScore;
        bestMove = move;
      }
    }

    return bestMove;
  }
}