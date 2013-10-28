package org.ggp.base.util.propnet.polymorphic;

import org.ggp.base.util.propnet.architecture.Component;

//Base class for all non-abstract And implementations.  Need so that
//instanceof PolymorphicNot can be used regardless of the concrete class hierarchy produced by the factory
public abstract interface PolymorphicNot extends PolymorphicComponent {

}
