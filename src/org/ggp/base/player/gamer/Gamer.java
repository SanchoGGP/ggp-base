
package org.ggp.base.player.gamer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ggp.base.apps.player.config.ConfigPanel;
import org.ggp.base.apps.player.config.EmptyConfigPanel;
import org.ggp.base.apps.player.detail.DetailPanel;
import org.ggp.base.apps.player.detail.EmptyDetailPanel;
import org.ggp.base.player.gamer.exception.AbortingException;
import org.ggp.base.player.gamer.exception.GamePreviewException;
import org.ggp.base.player.gamer.exception.MetaGamingException;
import org.ggp.base.player.gamer.exception.MoveSelectionException;
import org.ggp.base.player.gamer.exception.StoppingException;
import org.ggp.base.player.gamer.statemachine.sancho.RuntimeGameCharacteristics;
import org.ggp.base.util.game.GDLTranslator;
import org.ggp.base.util.game.Game;
import org.ggp.base.util.gdl.grammar.GdlConstant;
import org.ggp.base.util.gdl.grammar.GdlTerm;
import org.ggp.base.util.match.Match;
import org.ggp.base.util.observer.Event;
import org.ggp.base.util.observer.Observer;
import org.ggp.base.util.observer.Subject;
import org.ggp.base.util.symbol.grammar.Symbol;


/**
 * The Gamer class defines methods for both meta-gaming and move selection in a
 * pre-specified amount of time. The Gamer class is based on the
 * <i>algorithm</i> design pattern.
 */
public abstract class Gamer implements Subject
{
  private static final Logger LOGGER = LogManager.getLogger();

  private   Match                      match;
  private   GdlConstant                roleName;
  private   int                        port;
  private   GDLTranslator              mGDLTranslator;
  protected RuntimeGameCharacteristics mGameCharacteristics;

  public Gamer()
  {
    observers = new ArrayList<>();

    // When not playing a match, the variables 'match'
    // and 'roleName' should be NULL. This indicates that
    // the player is available for starting a new match.
    match = null;
    roleName = null;
  }

  /*
   * The following values are recommendations to the implementations for the
   * minimum length of time to leave between the stated timeout and when you
   * actually return from metaGame and selectMove. They are stored here so they
   * can be shared amongst all Gamers.
   */
  public static final long PREFERRED_METAGAME_BUFFER = 3900;
  public static final long PREFERRED_PLAY_BUFFER     = 1900;

  // ==== The Gaming Algorithms ====
  public abstract void metaGame(long timeout) throws MetaGamingException;

  public abstract GdlTerm selectMove(long timeout)
      throws MoveSelectionException;

  /*
   * Note that the match's goal values will not necessarily be known when
   * stop() is called, as we only know the final set of moves and haven't
   * interpreted them yet. To get the final goal values, process the final
   * moves of the game.
   */
  public abstract void stop() throws StoppingException; // Cleanly stop playing the match

  public abstract void abort() throws AbortingException; // Abruptly stop playing the match

  public abstract void preview(Game g, long timeout)
      throws GamePreviewException; // Preview a game

  // ==== Gamer Profile and Configuration ====
  public abstract String getName();

  public String getSpecies()
  {
    return null;
  }

  public boolean isComputerPlayer()
  {
    return true;
  }

  public ConfigPanel getConfigPanel()
  {
    return new EmptyConfigPanel();
  }

  /**
   * Setter
   * @param thePort Note the port this gamer is playing on
   */
  public void notePort(int thePort)
  {
    port = thePort;
  }

  /**
   * Getter
   * @return the port this gamer is playing on
   */
  public int getPort()
  {
    return port;
  }

  /**
   * Configure the player.
   *
   * Players wishing to receive command-line configuration should override this
   * method.
   *
   * @param xiParamIndex - index of the parameter (starting at 0).
   * @param xiParameter  - value of the parameter.
   */
  public void configure(int xiParamIndex, String xiParameter)
  {
    // Base implementation does nothing.
  }

  public DetailPanel getDetailPanel()
  {
    return new EmptyDetailPanel();
  }

  // ==== Accessors ====
  public final Match getMatch()
  {
    return match;
  }

  public final void setMatch(Match match)
  {
    this.match = match;
  }

  public final GdlConstant getRoleName()
  {
    return roleName;
  }

  public final void setRoleName(GdlConstant roleName)
  {
    this.roleName = roleName;
  }

  // ==== GDL Translation ====
  /**
   * @param xiGDLTranslator - the GDL translator
   */
  public void setGDLTranslator(GDLTranslator xiGDLTranslator)
  {
    mGDLTranslator = xiGDLTranslator;

    if (xiGDLTranslator != null)
    {
      // Abort games that would cause us to hang.
      File lPoison = new File(mGDLTranslator.getGameDir(), "poison");
      if (lPoison.exists())
      {
        LOGGER.error("Aborting poisoned game: " + lPoison.getPath());
        throw new RuntimeException("Aborting poisoned game: " + lPoison.getPath());
      }
      mGameCharacteristics = new RuntimeGameCharacteristics(mGDLTranslator.getGameDir());
    }
    else
    {
      mGameCharacteristics = null;
    }
  }

  public Symbol networkToInternal(Symbol xiNetworkSymbol)
  {
    return mGDLTranslator.networkToInternal(xiNetworkSymbol);
  }

  public Symbol internalToNetwork(Symbol xiInternalSymbol)
  {
    return mGDLTranslator.internalToNetwork(xiInternalSymbol);
  }

  public String getGameName()
  {
    return mGDLTranslator.getGameDir().getName();
  }

  public File getGameDir()
  {
    return mGDLTranslator.getGameDir();
  }

  // ==== Observer Stuff ====
  private final List<Observer> observers;

  @Override
  public final void addObserver(Observer observer)
  {
    observers.add(observer);
  }

  @Override
  public final void notifyObservers(Event event)
  {
    for (Observer observer : observers)
    {
      observer.observe(event);
    }
  }
}