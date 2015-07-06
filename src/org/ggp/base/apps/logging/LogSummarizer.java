
package org.ggp.base.apps.logging;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.ggp.base.util.http.HttpReader;
import org.ggp.base.util.http.HttpWriter;
import org.ggp.base.util.logging.LogSummaryGenerator;

/**
 * The "Exponent" Log Summarizer Server is a multi-threaded web server that
 * makes log summaries and sends them back to remote clients. These log
 * summaries should not contain any sensitive data; the summarizer can be
 * queried by anyone and its summaries are made publicly available on the
 * GGP.org viewer alongside the other information about each match. SAMPLE
 * INVOCATION (when running locally):
 * ResourceLoader.load_raw('http://127.0.0.1:9199/matchABC'); The Log
 * Summarizer Server replies with a JSON summary of the logs for "matchABC".
 *
 * @author Sam Schreiber
 */
public class LogSummarizer
{
  private static final int                 SERVER_PORT = 9199;
  private static final LogSummaryGenerator SUMMARY_GENERATOR = new LogSummaryGenerator();

  /**
   * Request handling thread for generating log summaries for upload to Tiltyard.
   */
  static class SummarizeLogThread extends Thread
  {
    private Socket mConnection;

    /**
     * Create a thread for handling a request.
     *
     * @param xiConnection - the connection from which to read the request.
     */
    public SummarizeLogThread(Socket xiConnection)
    {
      mConnection = xiConnection;
    }

    @Override
    public void run()
    {
      long lStartTime = System.currentTimeMillis();
      try
      {
        String lRequest = HttpReader.readRequestAsServer(mConnection);
        String lResponse;
        String lContentType;

        if (lRequest.equals("viz.html") || lRequest.endsWith(".js"))
        {
          lContentType = "text/html";
          StringBuffer lBuffer = new StringBuffer();
          List<String> lLines = Files.readAllLines(Paths.get("src_viz/" + lRequest), StandardCharsets.UTF_8);
          for (String lLine : lLines)
          {
            lBuffer.append(lLine);
            lBuffer.append('\n');
          }
          lResponse = lBuffer.toString();
        }
        else if (lRequest.startsWith("localview/"))
        {
          lContentType = "text/html";
          StringBuffer lBuffer = new StringBuffer();
          List<String> lLines = Files.readAllLines(Paths.get("src_viz/localview.html"), StandardCharsets.UTF_8);
          for (String lLine : lLines)
          {
            lBuffer.append(lLine);
            lBuffer.append('\n');
          }
          lResponse = lBuffer.toString();
        }
        else if (lRequest.equals("favicon.ico"))
        {
          // Ignore requests for favicon.
          lContentType = "text/html";
          lResponse = "";
        }
        else
        {
          System.out.println(System.currentTimeMillis() + ": Got request for logs for " + lRequest);
          lContentType = "text/acl";
          lResponse = SUMMARY_GENERATOR.getLogSummary(lRequest);
          System.out.println(System.currentTimeMillis() + ": Got " + lResponse.length() + " bytes of logs for " + lRequest);
          long lDuration = System.currentTimeMillis() - lStartTime;
          System.out.println(System.currentTimeMillis() + ": Took " + lDuration + "ms to generate the logs");
        }

        HttpWriter.writeAsServer(mConnection, lResponse, lContentType);
        mConnection.close();
        System.out.println(System.currentTimeMillis() + ": Successfully replied to log request for " + lRequest);
      }
      catch (IOException e)
      {
        e.printStackTrace();
        throw new RuntimeException(e);
      }

      // Prompt the JVM to do garbage collection.
      long lEndGCTime = System.currentTimeMillis() + 3000;
      for (int ii = 0; ii < 1000 && System.currentTimeMillis() < lEndGCTime; ii++)
      {
        System.gc();
        try {Thread.sleep(1);} catch (InterruptedException lEx) {/* Whatever */}
      }
    }
  }

  /**
   * Start the log summarizer.
   *
   * @param xiArgs - none.
   */
  @SuppressWarnings("resource")
  public static void main(String[] xiArgs)
  {
    ServerSocket listener = null;
    try
    {
      listener = new ServerSocket(SERVER_PORT);
    }
    catch (IOException e)
    {
      System.err.println("Could not open server on port " + SERVER_PORT + ": " + e);
      e.printStackTrace();
      return;
    }

    while (true)
    {
      Socket connection = null;

      try
      {
        connection = listener.accept();
      }
      catch (IOException lEx)
      {
        System.err.println("Failed to accept connection");
        lEx.printStackTrace();
      }

      if (connection != null)
      {
        Thread handlerThread = new SummarizeLogThread(connection);
        handlerThread.start();
      }
    }
  }
}