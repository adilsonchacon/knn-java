package br.poli.ecomp.rp;

import java.util.Comparator;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;

public class PairComparator implements Comparator<Pair<Double, Instance>> {

	public int compare(Pair<Double, Instance> o1, Pair<Double, Instance> o2) {
		return o1.getFirst().compareTo(o2.getFirst());
	}

}
