package org.ggp.base.player.gamer.statemachine.sancho;

import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility methods for logging statistics.
 */
public class StatsLogUtils
{
  /**
   * Graph types in the visualisation.
   */
  public static enum Graph
  {
    /**
     * Memory usage.
     */
    MEM,

    /**
     * GC time.
     */
    GC,

    /**
     * Performance.
     */
    PERF;
  }

  /**
   * Series types.
   */
  public static enum SeriesType
  {
    /**
     * Raw series.  Y-values plotted are the recorded y-values.
     */
    RAW,

    /**
     * Difference series.  Y-values plotted are the difference between the consecutive recorded y-values.  So, a series
     * with y-values 1,2,2,3,4 would be plotted as -,1,0,1,1.
     */
    DIFF,

    /**
     * Rate series.  Y-values plotted are the difference between the consecutive recorded y-values divided by the
     * difference between the corresponding x-values.
     */
    RATE;
  }

  /**
   * Statistics series for logging.
   */
  public static enum Series
  {
    /**
     * Currently used heap (bytes).
     */
    MEM_USED       (Graph.MEM,  0, SeriesType.RAW,  "Used"),

    /**
     * Memory allocated from the O.S. for heap usage (bytes).
     */
    MEM_COMMITTED  (Graph.MEM,  0, SeriesType.RAW,  "Committed"),

    /**
     * Maximum configured heap size (bytes).
     */
    MEM_MAX        (Graph.MEM,  0, SeriesType.RAW,  "Max"),

    /**
     * Garbage collection time (ms).
     */
    GC_TIME        (Graph.GC,   0, SeriesType.DIFF, "Time"),

    /**
     * Garbage collection count.
     */
    GC_COUNT       (Graph.GC,   1, SeriesType.DIFF, "Count"),

    /**
     * Node expansions.
     */
    NODE_EXPANSIONS(Graph.PERF, 0, SeriesType.RATE, "Expansions"),

    /**
     * Depth charges.
     */
    DEPTH_CHARGES  (Graph.PERF, 1, SeriesType.RATE, "Depth charges");

    private final Graph          mGraph;
    private final int            mAxis;
    private final SeriesType     mSeriesType;
    private final String         mName;
    private final Vector<String> mXValues;
    private final Vector<String> mYValues;

    private static final Pattern LINE_PATTERN = Pattern.compile("^([^,]+),(\\d+),(\\d+)");

    private Series(Graph xiGraph, int xiAxis, SeriesType xiSeriesType, String xiName)
    {
      mGraph      = xiGraph;
      mAxis       = xiAxis;
      mSeriesType = xiSeriesType;
      mName       = xiName;
      mXValues    = new Vector<>();
      mYValues    = new Vector<>();
    }

    /**
     * Log a data point.
     *
     * @param xiBuffer - the buffer to append to.
     * @param xiTime   - the x-value (usually a time in ms).
     * @param xiValue  - the y-value.
     */
    public void logDataPoint(StringBuffer xiBuffer, long xiTime, long xiValue)
    {
      xiBuffer.append(this);
      xiBuffer.append(',');
      xiBuffer.append(xiTime);
      xiBuffer.append(',');
      xiBuffer.append(xiValue);
      xiBuffer.append('\n');
    }

    /**
     * Load a data point into this series from file.
     *
     * For use by the log summariser only.
     *
     * @param lLine - the line in the file.
     */
    public static void loadDataPoint(String lLine)
    {
      Matcher lMatcher = LINE_PATTERN.matcher(lLine);
      if (lMatcher.matches())
      {
        Series lSeries = Series.valueOf(lMatcher.group(1));
        lSeries.addDataPoint(lMatcher.group(2), lMatcher.group(3));
      }
    }

    private void addDataPoint(String xiXValue, String xiYValue)
    {
      assert(mXValues.size() == mYValues.size()) : mXValues.size() + " X-values but " + mYValues.size() + " Y-values";

      mXValues.add(xiXValue);
      mYValues.add(xiYValue);
    }

    /**
     * Append details of this Series to a JSON buffer.
     *
     * For use by the log summariser only.
     *
     * @param xiBuffer - the buffer.
     */
    public void appendToJSON(StringBuffer xiBuffer)
    {
      assert(mXValues.size() == mYValues.size()) : mXValues.size() + " X-values but " + mYValues.size() + " Y-values";

      xiBuffer.append("{\"showon\":\"");
      xiBuffer.append(mGraph);
      xiBuffer.append("\",\"type\":\"line\",\"name\":\"");
      xiBuffer.append(mName);
      xiBuffer.append("\",\"yAxis\":");
      xiBuffer.append(mAxis);
      xiBuffer.append(",\"data\":[");
      for (int lii = 0; lii < mXValues.size(); lii++)
      {
        xiBuffer.append('[');
        xiBuffer.append(mXValues.get(lii));
        xiBuffer.append(',');
        xiBuffer.append(mYValues.get(lii));
        xiBuffer.append(']');
        xiBuffer.append(',');
      }
      xiBuffer.setLength(xiBuffer.length() - 1);
      xiBuffer.append("]}");
    }

    /**
     * @return whether the series is empty.
     *
     * For use by the log summariser only.
     */
    public boolean isEmpty()
    {
      assert(mXValues.size() == mYValues.size()) : mXValues.size() + " X-values but " + mYValues.size() + " Y-values";

      return mXValues.isEmpty();
    }

    /**
     * Clear all data points from the series.
     *
     * For use by the log summariser only.
     */
    public void reset()
    {
      mXValues.clear();
      mYValues.clear();

      assert(mXValues.size() == 0) : "X-values not cleared";
      assert(mYValues.size() == 0) : "Y-values not cleared";
    }

  }
}
