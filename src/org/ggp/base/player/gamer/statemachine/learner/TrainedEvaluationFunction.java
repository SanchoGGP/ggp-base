package org.ggp.base.player.gamer.statemachine.learner;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.util.OpenBitSet;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * A generic position evaluation function, based on neural networks.
 */
public class TrainedEvaluationFunction implements EvaluationFunction
{
  private static final Logger LOGGER = LogManager.getLogger();

  public final double INITIAL_LEARNING_RATE;

  private final int mInputSize;
  private final int mOutputSize;
  private final boolean m2PlayerFixedSum;
  private final NeuralNetwork<BackPropagation> mNetwork;
  private final BackPropagation mLearningRule;

  private final DataSet mTrainingSet;
  HashMap<ForwardDeadReckonInternalMachineState, double[]> mTrainingData = new HashMap<>();
  TObjectIntHashMap<ForwardDeadReckonInternalMachineState> mTrainingCount = new TObjectIntHashMap<>();

  private static final boolean TTT_MANUAL_FEATURES = true;

  public TrainedEvaluationFunction(int xiInputSize, int xiOutputSize, boolean xi2PlayerFixedSum)
  {
    if (TTT_MANUAL_FEATURES) xiInputSize += 8;

    // Create a neural network.
    m2PlayerFixedSum = xi2PlayerFixedSum;
    mInputSize = xiInputSize;
    mOutputSize = xi2PlayerFixedSum ? 1 : xiOutputSize;

    TransferFunctionType lTransferFunction = TransferFunctionType.SIGMOID;
    INITIAL_LEARNING_RATE = 0.05;
    double lInitialWeightMax = 1 / Math.sqrt(mInputSize);
    double lInitialWeightMin = -lInitialWeightMax;

    mNetwork = new MultiLayerPerceptron(lTransferFunction,
                                        mInputSize,     // Input layer, 1 neuron per base proposition
                                        mInputSize * 3, // Hidden layer(s)
                                        mOutputSize);   // Output layer, 1 neuron per role (except for fixed sum,
                                                        // where we only need 1).
    mNetwork.randomizeWeights(lInitialWeightMin, lInitialWeightMax);

    // Create a training set.
    mTrainingSet = new DataSet(mInputSize, mOutputSize);

    // Create a learning rule.
    mLearningRule = createLearningRule();

    LOGGER.info("Created an evaluation function with " + mNetwork.getInputsCount() + " inputs & " +
                mNetwork.getOutputsCount() + " outputs");
  }

  @SuppressWarnings("unchecked")
  public TrainedEvaluationFunction(String xiFilename, boolean xi2PlayerFixedSum)
  {
    // Load the neural network from disk.
    mNetwork = NeuralNetwork.createFromFile(xiFilename);
    mInputSize = mNetwork.getInputsCount();
    mOutputSize = mNetwork.getOutputsCount();
    m2PlayerFixedSum = xi2PlayerFixedSum;

    LOGGER.info("Reloaded an evaluation function with " + mNetwork.getInputsCount() + " inputs, " +
                mNetwork.getLayerAt(1).getNeuronsCount() + " hidden units & " +
                mNetwork.getOutputsCount() + " outputs");

    // Create a training set.
    mTrainingSet = new DataSet(mInputSize, mOutputSize);

    // Create a learning rule.
    mLearningRule = createLearningRule();

    INITIAL_LEARNING_RATE = 0.001; // !! ARR Pick a sensible value.  Dependent on unit used in saved file.
  }

  private BackPropagation createLearningRule()
  {
    BackPropagation lLearningRule = new BackPropagation();
    lLearningRule.setMaxIterations(1);
    lLearningRule.setLearningRate(INITIAL_LEARNING_RATE);
    lLearningRule.setNeuralNetwork(mNetwork);
    return lLearningRule;
  }

  @Override
  public double[] evaluate(ForwardDeadReckonInternalMachineState xiState)
  {
    double[] lInputs = convertStateToInputs(xiState);
    mNetwork.setInput(lInputs);
    mNetwork.calculate();

    double[] lOutputs;
    if (m2PlayerFixedSum)
    {
      lOutputs = new double[2];
      lOutputs[0] = mNetwork.getOutput()[0] * 100;
      lOutputs[1] = 100 - lOutputs[0];
    }
    else
    {
      lOutputs = mNetwork.getOutput().clone();
      for (int lii = 0; lii < mOutputSize; lii++)
      {
        lOutputs[lii] *= 100;
      }
    }
    return lOutputs;
  }

  @Override
  public void sample(ForwardDeadReckonInternalMachineState xiState,
                     double[] xiGoalValues)
  {
    assert((xiGoalValues.length == mOutputSize) || (m2PlayerFixedSum));

    // If this is a zero-sum game, only train from the first player's goal.  It's considerably more efficient than
    // trying to train both.
    if ((xiGoalValues.length == 2) && (m2PlayerFixedSum))
    {
      xiGoalValues = new double[] {xiGoalValues[0]};
    }

    // Copy the state.  The one we've been passed will be reused.
    ForwardDeadReckonInternalMachineState lState = new ForwardDeadReckonInternalMachineState(xiState);

    // Storing the training example.  Later samples (for the same state) override earlier ones.
    mTrainingData.put(lState, xiGoalValues);

    // Increment the count.
    if (mTrainingCount.containsKey(lState))
    {
      mTrainingCount.put(lState, mTrainingCount.get(lState) + 1);
    }
    else
    {
      mTrainingCount.put(lState, 1);
    }
  }

  @Override
  public void clearSamples()
  {
    mTrainingData.clear();
    mTrainingCount.clear();
  }

  private double[] convertStateToInputs(ForwardDeadReckonInternalMachineState xiState)
  {
    double[] lInputs = new double[mInputSize];
    int lNumProps = mInputSize;
    if (TTT_MANUAL_FEATURES) lNumProps -= 8;
    OpenBitSet lState = xiState.getContents();
    for (int lii = 0; lii < lNumProps; lii++)
    {
      lInputs[lii] = lState.fastGet(lii + xiState.firstBasePropIndex) ? 1 : -1;
    }

    if (TTT_MANUAL_FEATURES)
    {
      int[][] lLines = {{1, 1, 1, 2, 1, 3},
                        {2, 1, 2, 2, 2, 3},
                        {3, 1, 3, 2, 3, 3},
                        {1, 1, 2, 1, 3, 1},
                        {1, 2, 2, 2, 3, 2},
                        {1, 3, 2, 3, 3, 3},
                        {1, 1, 2, 2, 3, 3},
                        {3, 1, 2, 2, 1, 3}};

      for (int lii = 0; lii < 8; lii++)
      {
        lInputs[lNumProps + lii] = scoreLine(lState, lLines[lii]);
      }
    }

    return lInputs;
  }

  private int scoreLine(OpenBitSet xiState, int[] xiLine)
  {
    int lScore = 0;

    for (int lCell = 0; lCell < 6; lCell += 2)
    {
      int lX = xiLine[lCell] - 1;
      int lY = xiLine[lCell + 1] - 1;

      int lCellIndex = (lX * 9) + (lY * 3);
      int lPlayer1PropIndex = lCellIndex + 2; // X-player (blank 1st, then o-player)
      int lPlayer2PropIndex = lCellIndex + 1; // O-player

      if (xiState.fastGet(lPlayer1PropIndex)) lScore++;
      if (xiState.fastGet(lPlayer2PropIndex)) lScore--;
    }

    return lScore;
  }

  private double[] normaliseOutputs(double[] lGDLOutputs)
  {
    double[] lOutputs = new double[mOutputSize];
    for (int lii = 0; lii < mOutputSize; lii++)
    {
      lOutputs[lii] = lGDLOutputs[lii] / 100.0;
    }
    return lOutputs;
  }

  /**
   * Train the network with the current training data (from calls to sample()).
   *
   * @return the average training error from the batch.
   */
  public double train()
  {
    TObjectIntHashMap<ForwardDeadReckonInternalMachineState> lLastRequiredIteration = new TObjectIntHashMap<>();

    StringBuilder lXs = new StringBuilder();
    StringBuilder lYs = new StringBuilder();

    mTrainingSet.clear();
    for (Entry<ForwardDeadReckonInternalMachineState, double[]> lEntry : mTrainingData.entrySet())
    {
      ForwardDeadReckonInternalMachineState lState = lEntry.getKey();

      // Add additional weight to states that have been seen often.
      // for (int lii = 0; lii < mTrainingCount.get(lState); lii++)
      {
        double[] lInputs = convertStateToInputs(lState);
        double[] lOutputs = normaliseOutputs(lEntry.getValue());
        mTrainingSet.addRow(lInputs, lOutputs);

        lXs.append("X_train = np.append(X_train, [");
        lXs.append(Arrays.toString(lInputs));
        lXs.append("], axis=0)\n");

        lYs.append("Y_train = np.append(Y_train, ");
        lYs.append(Arrays.toString(lOutputs));
        lYs.append(")\n");
      }

      lLastRequiredIteration.put(lState, 0);
    }

    lXs.setLength(lXs.length() - 2);
    lYs.setLength(lYs.length() - 1);
    LOGGER.info(lXs);
    LOGGER.info(lYs);

    // !! ARR Test code to do lots of iterations on a single sample set.  Useful when doing 9-ply TTT.
    double lTrainingErr = 0;
    int lNumBonusIterations = 100000;
    double lLimit = 3.0;
    for (int lii = 0; lii < lNumBonusIterations; lii++)
    {
      if (lii == 4000) mLearningRule.setLearningRate(mLearningRule.getLearningRate() * 0.5);

      if (lii % 10 == 0)
      {
        double lRealErr = 0;
        double lMaxErr = 0;
        mTrainingSet.clear();
        for (Entry<ForwardDeadReckonInternalMachineState, double[]> lEntry : mTrainingData.entrySet())
        {
          ForwardDeadReckonInternalMachineState lState = lEntry.getKey();
          double lSampleErr = Math.abs(evaluate(lState)[0] - lEntry.getValue()[0]);
          lRealErr += lSampleErr;
          lMaxErr = Math.max(lMaxErr, lSampleErr);

          // Only train samples with a "large" error or those that have recently had a large error.
          if ((lSampleErr > lLimit) || (lLastRequiredIteration.get(lState) > lii - 100))
          {
            mTrainingSet.addRow(convertStateToInputs(lState),
                                normaliseOutputs(lEntry.getValue()));

            if (lSampleErr > lLimit)
            {
              lLastRequiredIteration.put(lState, lii);
            }
          }
        }
        lRealErr /= mTrainingData.size();

        LOGGER.info("Iteration " + lii +
                    ", training err " + ((lTrainingErr * 100.0) / 10.0) +
                    ", real err = " + lRealErr +
                    ", max err = " + lMaxErr +
                    ", samples in set = " + mTrainingSet.size());
        lTrainingErr = 0;

        if (mTrainingSet.size() == 0)
        {
          LOGGER.info("Training complete");
          LOGGER.info("Weights: " + Arrays.toString(mNetwork.getWeights()));
          mNetwork.save("complete.nnet");
          System.exit(1);
        }
      }
      mLearningRule.doOneLearningIteration(mTrainingSet);
      lTrainingErr += mLearningRule.getErrorFunction().getTotalError();
    }

    // Train the network.
    mLearningRule.doOneLearningIteration(mTrainingSet);
    return mLearningRule.getErrorFunction().getTotalError();
  }

  public void replaceWith(TrainedEvaluationFunction xiSource)
  {
    mNetwork.setWeights(xiSource.mNetwork.getWeights());
  }

  public double cool()
  {
    double lNewRate = mLearningRule.getLearningRate() * 0.9999;
    mLearningRule.setLearningRate(lNewRate);
    return lNewRate;
  }

  public void save()
  {
    mNetwork.save("Game.nnet");
  }
}