
package org.ggp.base.player.gamer.statemachine;

import java.util.ArrayList;
import java.util.List;

import org.ggp.base.player.gamer.Gamer;
import org.ggp.base.player.gamer.exception.AbortingException;
import org.ggp.base.player.gamer.exception.MetaGamingException;
import org.ggp.base.player.gamer.exception.MoveSelectionException;
import org.ggp.base.player.gamer.exception.StoppingException;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.logging.GamerLogger;
import org.ggp.base.util.statemachine.MachineState;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.Role;
import org.ggp.base.util.statemachine.StateMachine;
import org.ggp.base.util.statemachine.exceptions.GoalDefinitionException;
import org.ggp.base.util.statemachine.exceptions.MoveDefinitionException;
import org.ggp.base.util.statemachine.exceptions.TransitionDefinitionException;


/**
 * The base class for Gamers that rely on representing games as state machines.
 * Almost every player should subclass this class, since it provides the common
 * methods for interpreting the match history as transitions in a state
 * machine, and for keeping an up-to-date view of the current state of the
 * game. See @SimpleSearchLightGamer, @HumanGamer, and @RandomGamer for
 * examples.
 *
 * @author evancox
 * @author Sam
 */
public abstract class StateMachineGamer extends Gamer
{
  private long metaGamingTimeout = 0;

  // =====================================================================
  // First, the abstract methods which need to be overriden by subclasses.
  // These determine what state machine is used, what the gamer does during
  // metagaming, and how the gamer selects moves.

  /**
   * Defines which state machine this gamer will use.
   *
   * @return
   */
  public abstract StateMachine getInitialStateMachine();

  /**
   * Defines the metagaming action taken by a player during the START_CLOCK
   *
   * @param timeout
   *          time in milliseconds since the era when this function must return
   * @throws TransitionDefinitionException
   * @throws MoveDefinitionException
   * @throws GoalDefinitionException
   */
  public abstract void stateMachineMetaGame(long timeout)
      throws TransitionDefinitionException, MoveDefinitionException,
      GoalDefinitionException;

  /**
   * Defines the algorithm that the player uses to select their move.
   *
   * @param timeout
   *          time in milliseconds since the era when this function must return
   * @return Move - the move selected by the player
   * @throws TransitionDefinitionException
   * @throws MoveDefinitionException
   * @throws GoalDefinitionException
   */
  public abstract Move stateMachineSelectMove(long timeout)
      throws TransitionDefinitionException, MoveDefinitionException,
      GoalDefinitionException;

  /**
   * Defines any actions that the player takes upon the game cleanly ending.
   */
  public abstract void stateMachineStop();

  /**
   * Defines any actions that the player takes upon the game abruptly ending.
   */
  public abstract void stateMachineAbort();

  // =====================================================================
  // Next, methods which can be used by subclasses to get information about
  // the current state of the game, and tweak the state machine on the fly.

  public final long getMetaGamingTimeout()
  {
    return metaGamingTimeout;
  }

  /**
   * Returns the current state of the game.
   */
  public final MachineState getCurrentState()
  {
    return currentState;
  }

  /**
   * Returns the role that this gamer is playing as in the game.
   */
  public final Role getRole()
  {
    return role;
  }

  /**
   * Returns the state machine. This is used for calculating the next state and
   * other operations, such as computing the legal moves for all players,
   * whether states are terminal, and the goal values of terminal states.
   */
  public final StateMachine getStateMachine()
  {
    return stateMachine;
  }

  /**
   * Cleans up the role, currentState and stateMachine. This should only be
   * used when a match is over, and even then only when you really need to free
   * up resources that the state machine has tied up. Currently, it is only
   * used in the Proxy, for players designed to run 24/7.
   */
  protected final void cleanupAfterMatch()
  {
    role = null;
    currentState = null;
    stateMachine = null;
    setMatch(null);
    setRoleName(null);
  }

  /**
   * Switches stateMachine to newStateMachine, playing through the match
   * history to the current state so that currentState is expressed using a
   * MachineState generated by the new state machine. This is not done in a
   * thread-safe fashion with respect to the rest of the gamer, so be careful
   * when using this method.
   *
   * @param newStateMachine
   *          the new state machine
   */
  protected final void switchStateMachine(StateMachine newStateMachine)
  {
    try
    {
      MachineState newCurrentState = newStateMachine.getInitialState();
      Role newRole = newStateMachine.getRoleFromConstant(getRoleName());

      // Attempt to run through the game history in the new machine
      List<List<GdlTerm>> theMoveHistory = getMatch().getMoveHistory();
      for (List<GdlTerm> nextMove : theMoveHistory)
      {
        List<Move> theJointMove = new ArrayList<Move>();
        for (GdlTerm theSentence : nextMove)
          theJointMove.add(newStateMachine.getMoveFromTerm(theSentence));
        newCurrentState = newStateMachine
            .getNextStateDestructively(newCurrentState, theJointMove);
      }

      // Finally, switch over if everything went well.
      role = newRole;
      currentState = newCurrentState;
      stateMachine = newStateMachine;
    }
    catch (Exception e)
    {
      GamerLogger.log("GamePlayer",
                      "Caught an exception while switching state machine!");
      GamerLogger.logStackTrace("GamePlayer", e);
    }
  }

  // =====================================================================
  // Finally, methods which are overridden with proper state-machine-based
  // semantics. These basically wrap a state-machine-based view of the world
  // around the ordinary metaGame() and selectMove() functions, calling the
  // new stateMachineMetaGame() and stateMachineSelectMove() functions after
  // doing the state-machine-related book-keeping.

  /**
   * A wrapper function for stateMachineMetaGame. When the match begins, this
   * initializes the state machine and role using the match description, and
   * then calls stateMachineMetaGame.
   */
  @Override
  public final void metaGame(long timeout) throws MetaGamingException
  {
    try
    {
      metaGamingTimeout = timeout;

      stateMachine = getInitialStateMachine();
      stateMachine.initialize(getMatch().getGame().getRules());
      currentState = stateMachine.getInitialState();
      role = stateMachine.getRoleFromConstant(getRoleName());
      getMatch().appendState(currentState.getContents());

      stateMachineMetaGame(timeout);
    }
    catch (Exception e)
    {
      GamerLogger.logStackTrace("GamePlayer", e);
      throw new MetaGamingException(e);
    }
  }

  /**
   * A wrapper function for stateMachineSelectMove. When we are asked to select
   * a move, this advances the state machine up to the current state and then
   * calls stateMachineSelectMove to select a move based on that current state.
   */
  @Override
  public final GdlTerm selectMove(long timeout) throws MoveSelectionException
  {
    try
    {
      stateMachine.doPerMoveWork();

      List<GdlTerm> lastMoves = getMatch().getMostRecentMoves();
      if (lastMoves != null)
      {
        List<Move> moves = new ArrayList<Move>();
        for (GdlTerm sentence : lastMoves)
        {
          moves.add(stateMachine.getMoveFromTerm(sentence));
        }

        currentState = stateMachine.getNextState(currentState, moves);
        getMatch().appendState(currentState.getContents());
      }

      return stateMachineSelectMove(timeout).getContents();
    }
    catch (Exception e)
    {
      GamerLogger.logStackTrace("GamePlayer", e);
      throw new MoveSelectionException(e);
    }
  }

  @Override
  public void stop() throws StoppingException
  {
    try
    {
      stateMachine.doPerMoveWork();

      List<GdlTerm> lastMoves = getMatch().getMostRecentMoves();
      if (lastMoves != null)
      {
        List<Move> moves = new ArrayList<Move>();
        for (GdlTerm sentence : lastMoves)
        {
          moves.add(stateMachine.getMoveFromTerm(sentence));
        }

        currentState = stateMachine.getNextState(currentState, moves);
        getMatch().appendState(currentState.getContents());
        getMatch().markCompleted(stateMachine.getGoals(currentState));
      }

      stateMachineStop();
    }
    catch (Exception e)
    {
      GamerLogger.logStackTrace("GamePlayer", e);
      throw new StoppingException(e);
    }
  }

  @Override
  public void abort() throws AbortingException
  {
    try
    {
      stateMachineAbort();
    }
    catch (Exception e)
    {
      GamerLogger.logStackTrace("GamePlayer", e);
      throw new AbortingException(e);
    }
  }

  // Internal state about the current state of the state machine.
  private Role         role;
  private MachineState currentState;
  private StateMachine stateMachine;
}