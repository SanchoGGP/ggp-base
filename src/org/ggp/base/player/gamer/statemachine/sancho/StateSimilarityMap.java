package org.ggp.base.player.gamer.statemachine.sancho;

import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonPropNet;

public class StateSimilarityMap
{
  private class StateSimilarityBucket
  {
    private final int capacity = 4;

    public final TreeNodeRef[] refs = new TreeNodeRef[capacity];
    public int  size = 0;

    public void addNode(TreeNodeRef nodeRef)
    {
      // Check if this node is already present.  If not, no need to store.
      for (int i = 0; i < size; i++)
      {
        if (nodeRef.hasSameReferand(refs[i]))
        {
          //  Already present
          return;
        }
      }

      if (size < capacity)
      {
        // We still have space available.  Just store the reference.
        refs[size++] = nodeRef;
      }
      else
      {
        int evictee = -1;

        TreeNode lNodeToAdd = nodeRef.get();
        assert(lNodeToAdd != null);

        double highestEvictionMeasure = -Math.log(lNodeToAdd.numVisits + 1);

        for (int i = 0; i < capacity; i++)
        {
          TreeNodeRef ref = refs[i];
          TreeNode lNode = ref.get();
          if (lNode == null)
          {
            //  Effectively a free slot - no loss to evict it
            evictee = i;
            break;
          }

          double evictionMeasure = -Math.log(lNode.numVisits + 1);

          if ( evictionMeasure > highestEvictionMeasure )
          {
            evictionMeasure = highestEvictionMeasure;
            evictee = i;
          }
        }

        //  If the cache contained something less useful than the new entry replace it
        if ( evictee != -1 )
        {
          refs[evictee] = nodeRef;
        }
      }
    }
  }

  final private StateSimilarityBucket[] buckets;
  final private StateSimilarityHashGenerator hashGenerator;
  final private int maxMovesConsidered = 64;
  final private ForwardDeadReckonLegalMoveInfo[] moveBuffer = new ForwardDeadReckonLegalMoveInfo[maxMovesConsidered];
  final private double[] moveValueBuffer = new double[maxMovesConsidered];
  final private double[] moveWeightBuffer = new double[maxMovesConsidered];
  final private double[] topValues = new double[maxMovesConsidered];
  final private double[] topWeights = new double[maxMovesConsidered];
  private int numMovesBuffered;

  public StateSimilarityMap(ForwardDeadReckonPropNet propNet)
  {
    hashGenerator = new StateSimilarityHashGenerator(propNet);
    buckets = new StateSimilarityBucket[1<<StateSimilarityHashGenerator.hashSize];
  }

  public void add(TreeNodeRef nodeRef)
  {
    int hash = hashGenerator.getHash(nodeRef.get().state);

    if (buckets[hash] == null)
    {
      buckets[hash] = new StateSimilarityBucket();
    }
    buckets[hash].addNode(nodeRef);
  }

  public int getScoreEstimate(ForwardDeadReckonInternalMachineState state, double[] result)
  {
    for(int i = 0; i < result.length; i++)
    {
      result[i] = 0;
    }

    int hash = hashGenerator.getHash(state);

    StateSimilarityBucket bucket = buckets[hash];
    if ( bucket != null )
    {
      double totalWeight = 0;

      for(int i = 0; i < bucket.size; i++)
      {
        TreeNodeRef nodeRef = bucket.refs[i];
        TreeNode lNode = nodeRef.get();

        if (lNode != null && lNode.numVisits > 0 && state != lNode.state)
        {
          double distanceWeight = (1 - state.distance(lNode.state));
          double weight = distanceWeight*distanceWeight*Math.log(lNode.numVisits+1);

          for(int j = 0; j < result.length; j++)
          {
            result[j] += lNode.averageScores[j]*weight;
            assert(!Double.isNaN(result[j]));
          }

          totalWeight += weight;
        }
      }

      if ( totalWeight > 0 )
      {
        for(int i = 0; i < result.length; i++)
        {
          result[i] /= totalWeight;
          assert(!Double.isNaN(result[i]));
        }
      }
      return (int)(totalWeight);
    }

    return 0;
  }

  private TreeNode getJointMoveParent(TreeNode moveRoot, ForwardDeadReckonLegalMoveInfo[] partialJointMove)
  {
    int index = 0;
    TreeNode result = null;

    if ( partialJointMove[partialJointMove.length-1] != null )
    {
      return moveRoot;
    }

    while(index < partialJointMove.length && partialJointMove[index] != null)
    {
      if ( moveRoot.children == null )
      {
        return null;
      }

      boolean childFound = false;
      for(Object child : moveRoot.children)
      {
        ForwardDeadReckonLegalMoveInfo targetPartialMove = partialJointMove[index];
        TreeEdge childEdge = (child instanceof TreeEdge ? (TreeEdge)child : null);
        if ( child == targetPartialMove || (childEdge != null && childEdge.partialMove == targetPartialMove))
        {
          childFound = true;

          if (childEdge != null &&
              childEdge.child != null &&
              childEdge.child.getLive() != null &&
              childEdge.child.get().children != null)
          {
            result = childEdge.child.get();
            moveRoot = result;
            index++;
            break;
          }

          return null;
        }
      }

      if ( !childFound )
      {
        return null;
      }
    }

    return result;
  }

  private int getMoveSlot(ForwardDeadReckonLegalMoveInfo move)
  {
    for(int i = 0; i < numMovesBuffered; i++)
    {
      if ( moveBuffer[i] == move )
      {
        return i;
      }
    }

    if ( numMovesBuffered < maxMovesConsidered )
    {
      numMovesBuffered++;
    }

    moveBuffer[numMovesBuffered-1] = move;
    moveWeightBuffer[numMovesBuffered-1] = 0;
    moveValueBuffer[numMovesBuffered-1] = 0;

    return numMovesBuffered-1;
  }

  public int getTopMoves(ForwardDeadReckonInternalMachineState state, ForwardDeadReckonLegalMoveInfo[] partialJointMove, ForwardDeadReckonLegalMoveInfo[] result)
  {
    int hash = hashGenerator.getHash(state);

    numMovesBuffered = 0;

    int hammingCloseHash = hash;

    for(int nearbyHashIndex = 0; nearbyHashIndex <= StateSimilarityHashGenerator.hashSize; nearbyHashIndex++)
    {
      StateSimilarityBucket bucket = buckets[hammingCloseHash];
      if ( bucket != null )
      {
        for(int i = 0; i < bucket.size; i++)
        {
          TreeNodeRef nodeRef = bucket.refs[i];
          TreeNode lNode = nodeRef.get();

          if (lNode != null && lNode.numVisits > 0 && state != lNode.state)
          {
            double distanceWeight = (1 - state.distance(lNode.state));
            double weight = distanceWeight*distanceWeight*Math.log10(lNode.numVisits + 1);

            TreeNode node = getJointMoveParent(lNode, partialJointMove);
            if ( node != null && node.children != null )
            {
              for(Object child : node.children)
              {
                TreeEdge childEdge = (child instanceof TreeEdge ? (TreeEdge)child : null);
                if ( childEdge != null &&
                     childEdge.child != null &&
                     childEdge.child.getLive() != null &&
                     childEdge.child.get().numVisits > 0)
                {
                  TreeNode lChild = childEdge.child.get();
                  ForwardDeadReckonLegalMoveInfo move = childEdge.partialMove;
                  int moveSlotIndex = getMoveSlot(move);

                  double moveVal = weight*(lChild.averageScores[lChild.decidingRoleIndex]);

                  moveValueBuffer[moveSlotIndex] = (moveValueBuffer[moveSlotIndex]*moveWeightBuffer[moveSlotIndex] + moveVal)/(moveWeightBuffer[moveSlotIndex] + weight);
                  moveWeightBuffer[moveSlotIndex] += weight;
                }
              }
            }
          }
        }

        //  We look at all hashes within a Hamming distance of 1 from the original
        hammingCloseHash = hash ^ (1<<nearbyHashIndex);
      }

      //System.out.println("Found " + numMovesBuffered + " moves to buffer");
      int numTopMoves = 0;
      for(int i = 0; i < numMovesBuffered; i++)
      {
        int index = numTopMoves - 1;

        while( index >= 0 && moveValueBuffer[i] > topValues[index] )
        {
          index--;
        }

        if ( ++index < result.length )
        {
          for(int j = numTopMoves-1; j > index; j--)
          {
            topValues[j] = topValues[j-1];
            topWeights[j] = topWeights[j-1];

            result[j] = result[j-1];
          }

          topValues[index] = moveValueBuffer[i];
          topWeights[index] = moveWeightBuffer[i];
          result[index] = moveBuffer[i];

          if ( index == numTopMoves )
          {
            numTopMoves = index+1;
          }
        }
      }

      int i;
      double totalWeight = 0;
      double bestScore = topValues[0];
      final double ratioToBestCutoff = 0.8;

      for(i = 0; i < numTopMoves; i++)
      {
        if ( topValues[i] < ratioToBestCutoff*bestScore )
        {
          numTopMoves = i;
          break;
        }
        totalWeight += topWeights[i];
      }
      while(i < result.length)
      {
        result[i++] = null;
      }

      if ( numTopMoves > 0 )
      {
        totalWeight /= numTopMoves;
      }

      return (int)(totalWeight);
    }

    return 0;
  }
}
