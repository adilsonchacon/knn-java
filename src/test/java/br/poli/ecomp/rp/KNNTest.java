package br.poli.ecomp.rp;

import static org.junit.Assert.fail;

import java.io.IOException;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.junit.Test;

public class KNNTest {

	@Test
	public void readArff() {
		try {
			KNN knn = new KNN(2, new EuclideanDistance());
			knn.readArff(KNN.ARFF_FILE_PATH);
		} catch (IOException e) {
			fail(e.getMessage());
		}
	}

	@Test
	public void run() {
		try {
			KNN knn = new KNN(5, new EuclideanDistance());
			knn.run(KNN.ARFF_FILE_PATH);
			knn.printResult();
		} catch (IOException e) {
			fail(e.getMessage());
		}
	}
}
