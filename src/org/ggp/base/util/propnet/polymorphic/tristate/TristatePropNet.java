package org.ggp.base.util.propnet.polymorphic.tristate;

import org.ggp.base.util.propnet.polymorphic.PolymorphicComponent;
import org.ggp.base.util.propnet.polymorphic.PolymorphicPropNet;
import org.ggp.base.util.propnet.polymorphic.tristate.TristateComponent.Tristate;

public class TristatePropNet extends PolymorphicPropNet
{
  /**
   * Create a tri-state propnet from another propnet (of any kind).
   *
   * @param xiSourcePropnet - the source
   */
  public TristatePropNet(PolymorphicPropNet xiSourcePropnet)
  {
    super(xiSourcePropnet, new TristateComponentFactory());
  }

  /**
   * Reset the network to its default state.
   */
  public void reset()
  {
    // Reset all components.
    for (PolymorphicComponent lComponent : getComponents())
    {
      ((TristateComponent)lComponent).reset();
    }

    // Assume that the init proposition is false in all turns.  This means that we can't find latches which are only
    // rely on something happening during the first turn, but we can live with that.
    TristateProposition lInitProp = ((TristateProposition)getInitProposition());
    for (int lii = 0; lii < 3; lii++)
    {
      lInitProp.mState[lii].mValue = Tristate.FALSE;
      lInitProp.changeOutput(lii, false);
    }
  }
}
