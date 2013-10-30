package org.ggp.base.util.propnet.polymorphic.runtimeOptimized;

import org.ggp.base.util.gdl.grammar.GdlSentence;
import org.ggp.base.util.propnet.polymorphic.PolymorphicAnd;
import org.ggp.base.util.propnet.polymorphic.PolymorphicComponentFactory;
import org.ggp.base.util.propnet.polymorphic.PolymorphicConstant;
import org.ggp.base.util.propnet.polymorphic.PolymorphicNot;
import org.ggp.base.util.propnet.polymorphic.PolymorphicOr;
import org.ggp.base.util.propnet.polymorphic.PolymorphicProposition;
import org.ggp.base.util.propnet.polymorphic.PolymorphicTransition;

public class RuntimeOptimizedComponentFactory extends PolymorphicComponentFactory {

	public RuntimeOptimizedComponentFactory()
	{
	}
	
	@Override
	public PolymorphicAnd createAnd(int numInputs, int numOutputs) {
		return new RuntimeOptimizedAnd(numInputs, numOutputs);
	}
	
	@Override
	public PolymorphicOr createOr(int numInputs, int numOutputs) {
		return new RuntimeOptimizedOr(numInputs, numOutputs);
	}
	
	@Override
	public PolymorphicNot createNot(int numOutputs) {
		return new RuntimeOptimizedNot(numOutputs);
	}

	@Override
	public PolymorphicConstant createConstant(int numOutputs, boolean value)
	{
		return new RuntimeOptimizedConstant(numOutputs, value);
	}
	
	@Override
	public PolymorphicProposition createProposition(int numOutputs, GdlSentence name)
	{
		return new RuntimeOptimizedProposition(numOutputs, name);
	}

	@Override
	public PolymorphicTransition createTransition(int numOutputs)
	{
		return new RuntimeOptimizedTransition(numOutputs);
	}
}