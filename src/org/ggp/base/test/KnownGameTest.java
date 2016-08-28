package org.ggp.base.test;

import java.io.File;
import java.text.Collator;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import org.ggp.base.player.gamer.statemachine.StateMachineGamer;
import org.ggp.base.player.gamer.statemachine.sancho.MachineSpecificConfiguration;
import org.ggp.base.player.gamer.statemachine.sancho.MachineSpecificConfiguration.CfgItem;
import org.ggp.base.player.gamer.statemachine.sancho.RuntimeGameCharacteristics;
import org.ggp.base.player.request.factory.RequestFactory;
import org.ggp.base.player.request.factory.exceptions.RequestFormatException;
import org.ggp.base.util.game.CloudGameRepository;
import org.ggp.base.util.game.Game;
import org.ggp.base.util.game.GameRepository;
import org.ggp.base.util.http.HttpReader.GGPRequest;
import org.ggp.base.util.statemachine.Move;
import org.ggp.base.util.statemachine.StateMachine;
import org.ggp.base.util.statemachine.implementation.prover.ProverStateMachine;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Test puzzles.
 */
@RunWith(Parameterized.class)
public class KnownGameTest extends Assert
{
  private static HashSet<String> SKIP = new HashSet<>();
  static
  {
    SKIP.add("stanford.besthunter"); // Illegal GDL (Zero-arity relation should be a proposition).
    SKIP.add("stanford.firesheep");
    SKIP.add("stanford.go");          // Illegal GDL (unsafe rule).
    SKIP.add("stanford.kono");
    SKIP.add("stanford.madness");
    SKIP.add("stanford.pilgrimage");
    SKIP.add("stanford.skirmish"); // Covered by #385.
    SKIP.add("stanford.skirmishbad"); // Illegal GDL (unsafe rule).
    SKIP.add("stanford.untwistycomplex"); // Illegal GDL (missing "base"s).
    SKIP.add("stanford.yinsh");       // GDL refers to rule as if it were a proposition (using "true (order ...)").
    SKIP.add("stanford.zertz");       // Illegal GDL (unsafe rule).

    SKIP.add("stanford.arithmetic"); // GDL is so bugged that we refuse to play.  (No goals for the only role.)
    SKIP.add("stanford.arithmetic-stupid"); // Massive GDL means we refuse to learn.  Issue #154.
  }

  private static HashMap<String, String> AKA = new HashMap<>();
  static
  {
    AKA.put("stanford.chinesecheckers4local", "stanford.chinesecheckers4");
    AKA.put("stanford.jointconnectfour",      "stanford.dualconnectfour");
    AKA.put("stanford.threepuzzle",           "stanford.3puzzle");
    AKA.put("stanford.trifecta",              "stanford.tictactoe");
    AKA.put("stanford.rc6zero",               "stanford.reflectconnect6");
  }

  /**
   * Create a list of tests to run (1 per game) from the Stanford repository.
   *
   * @return the tests to run.
   */
  @Parameters(name="{0}")
  public static Iterable<? extends Object> data()
  {
    LinkedList<Object[]> lTests = new LinkedList<>();

    for (String lRepoName : new String[] {/*"base",*/ "stanford"})
    {
      // Get all the games in the repository.
      GameRepository lRepo = new CloudGameRepository(PuzzleBase.REPO_URL.get(lRepoName));

      // Filter them.
      for (String lGameName : lRepo.getGameKeys())
      {
        lTests.add(new Object[] {lRepoName + "." + lGameName, lRepo.getGame(lGameName)});
      }
    }

    // Sort the tests
    Collections.sort(lTests, new Comparator<Object[]>()
                     {
                       @Override
                       public int compare(Object[] xiA, Object[] xiB)
                       {
                         return Collator.getInstance().compare((String)(xiA[0]), (String)(xiB[0]));
                       }
                     });

    return lTests;
  }

  private final String mName;
  private final Game mGame;
  private final String mID;
  private final RequestFactory mRequestFactory;
  private final TestGamer mGamer;

  private boolean mStarted = false;

  /**
   * Create a test case for the specified game.
   *
   * @param xiName - the name of the game.
   * @param xiGame - the game.
   */
  public KnownGameTest(String xiName, Game xiGame)
  {
    mName = xiName;
    mGame = xiGame;
    mID = mName + "." + System.currentTimeMillis();

    // Create an instance of Sancho.
    mGamer = new TestGamer();
    mRequestFactory = new RequestFactory();

    // Make sure that learning is enabled for this test.
    MachineSpecificConfiguration.utOverrideCfgVal(CfgItem.DISABLE_LEARNING, false);
  }

  /**
   * Test that all games are recognised (i.e. we have a characteristic directory for them).
   *
   * @throws Exception
   */
  @Test
  public void testGameKnown() throws Exception
  {
    org.junit.Assume.assumeFalse(mName + " is broken", SKIP.contains(mName));

    sendStartRequest();

    // Check that we know about this game - and that it's recorded under the correct name.
    if (AKA.containsKey(mName))
    {
      // There are a few duplicate (or at least equivalent) games.  We only have 1 game directory for them, as recorded
      // in AKA above.
      assertEquals(AKA.get(mName), mGamer.getGameName());
    }
    else
    {
      if (mGamer.getGameName().matches("^1\\d+$"))
      {
        // New game.  Rename the directory (so, if run other than on CI, we'll pass next time) then assert anyway.
        File mWrongGameDir = mGamer.getGameDir();
        File mRightGameDir = new File(mWrongGameDir.getParentFile(), mName);

        if (mRightGameDir.exists())
        {
          assertTrue("Found game with same name but different GDL for " + mName, false);
        }
        else
        {
          mWrongGameDir.renameTo(mRightGameDir);
        }
      }
      assertEquals(mName, mGamer.getGameName());
    }

    // Check that we know the control set.
    assertNotNull("Control set not learned", mGamer.getCharacteristics().getControlMask());

    // Check that we know the number of factors.
    assertNotEquals("Factors not learned", 0, mGamer.getCharacteristics().getNumFactors());

    // Check that we have latch analysis results.
    // assertNotNull("Latches not learned", mGamer.getCharacteristics().getLatchesBasePositive());
  }

  /**
   * Abort any running game (which will happen if a test fails).
   *
   * @throws Exception
   */
  @After
  public void abortGame() throws Exception
  {
    // Abort the game (if running).
    if (mStarted)
    {
      getResponse("(abort " + mID + ")");
    }
  }

  /**
   * Send a start request for the current game to the test gamer.
   *
   * @throws RequestFormatException
   */
  private void sendStartRequest() throws RequestFormatException
  {
    // Extract game information.
    StateMachine lStateMachine = new ProverStateMachine();
    lStateMachine.initialize(mGame.getRules());
    String lRole = lStateMachine.getRoles()[0].toString();
    String lRules = mGame.getRulesheet();
    int lStartClock = 60;
    int lPlayClock = 60;

    // Send a start request to a test gamer.  That's enough to trigger game recognition and characteristic loading.
    String lRequest = "(start " +
                      mID + " " +
                      lRole + " " +
                      lRules + " " +
                      lStartClock + " " +
                      lPlayClock + " )";

    assertEquals("ready", getResponse(lRequest));
  }

  /**
   * Send a request to the player and get the response.
   *
   * @param xiRequest - the request to send
   * @return the response from the player.
   *
   * @throws RequestFormatException if the request was malformed.
   */
  private String getResponse(String xiRequest) throws RequestFormatException
  {
    GGPRequest lRequest = new GGPRequest();
    lRequest.mRequest = xiRequest;
    return mRequestFactory.create(mGamer, lRequest).process(System.currentTimeMillis());
  }

  /**
   * Dummy gamer.
   */
  public static class TestGamer extends StateMachineGamer
  {

    @Override
    public StateMachine getInitialStateMachine()
    {
      return new ProverStateMachine();
    }

    @Override
    public void stateMachineMetaGame(long xiTimeout)
    {
      // Do nothing.
    }

    @Override
    public Move stateMachineSelectMove(long xiTimeout)
    {
      // Do nothing.
      return null;
    }

    @Override
    public void stateMachineStop()
    {
      cleanupAfterMatch();
    }

    @Override
    public void stateMachineAbort()
    {
      cleanupAfterMatch();
    }

    @Override
    public void preview(Game xiGame, long xiTimeout)
    {
      // Do nothing.
    }

    @Override
    public String getName()
    {
      return "PuzzleBaseTestGamer";
    }

    /**
     * @return the game characteristics.
     */
    public RuntimeGameCharacteristics getCharacteristics()
    {
      return mGameCharacteristics;
    }
  }
}
