package org.ggp.base.player.gamer.statemachine.mctsref;

import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonInternalMachineState;
import org.ggp.base.util.propnet.polymorphic.forwardDeadReckon.ForwardDeadReckonLegalMoveInfo;

/**
 * Test-ground for #199.
 */
public class ARRSearchTreeNode extends SearchTreeNode<ARRSearchTree>
{
  private boolean mStopBackPropHere;

  public ARRSearchTreeNode(ARRSearchTree xiTree,
                           ForwardDeadReckonInternalMachineState xiState,
                           int xiChoosingRole)
  {
    super(xiTree, xiState, xiChoosingRole);
  }

  @Override
  protected SearchTreeNode<ARRSearchTree> select(ForwardDeadReckonLegalMoveInfo[] jointMove)
  {
    mStopBackPropHere = false;

    if (complete)
    {
      return this;
    }

    // Find the best child on exploitation alone.
    double lBestExploitationScore = -Double.MAX_VALUE;
    for(int i = 0; i < children.length; i++)
    {
      SearchTreeNode<ARRSearchTree> child = children[i];
      double lExploitationScore = lowerBound(child);

      if (lExploitationScore > lBestExploitationScore)
      {
        lBestExploitationScore = lExploitationScore;
      }
    }

    // Do the regular selection.
    SearchTreeNode<ARRSearchTree> lSelected = super.select(jointMove);

    // If the exploitation value of the selected child isn't sufficiently large, assume that it wouldn't be chosen and
    // block back-prop at this point.
    if (upperBound(lSelected) < lBestExploitationScore)
    {
      mStopBackPropHere = true;
    }

    return lSelected;
  }

  private double lowerBound(SearchTreeNode<ARRSearchTree> xiChild)
  {
    if (xiChild.numVisits == 0)
    {
      return 0;
    }

    return exploitationScore(xiChild) - (1 / Math.sqrt(4 * xiChild.numVisits));
  }

  private double upperBound(SearchTreeNode<ARRSearchTree> xiChild)
  {
    if (xiChild.numVisits == 0)
    {
      return 1;
    }

    return exploitationScore(xiChild) + (1 / Math.sqrt(4 * xiChild.numVisits));
  }

  @Override
  protected void updateScore(SearchTreeNode<ARRSearchTree> xiChild, double[] playoutResult)
  {
    if ((tree.mSuppressBackProp) || (mStopBackPropHere))
    {
      tree.mSuppressBackProp = true;
      return;
    }

    for(int i = 0; i < scoreVector.length; i++)
    {
      scoreVector[i] = (scoreVector[i]*numVisits + playoutResult[i])/(numVisits+1);
    }
  }

  @Override
  SearchTreeNode<ARRSearchTree> createNode(ForwardDeadReckonInternalMachineState xiState,
                                           int xiChoosingRole)
  {
    return new ARRSearchTreeNode(tree, xiState, xiChoosingRole);
  }
}
