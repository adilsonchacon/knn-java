package br.poli.ecomp.rp;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class KNN {

	private static Logger log = LoggerFactory.getLogger(KNN.class);

	/**
	 * If the dataset is not provided, uses the default dataset.
	 */
	public static final String ARFF_FILE_PATH = "/optdigits.arff";

	/**
	 * K nearest neighbors.
	 */
	private int K;

	/**
	 * The distance function to calculate the similarity.
	 */
	private DistanceMeasure distanceMeasure;

	/**
	 * The instances (patterns) loaded from the dataset.
	 */
	private Instances instances;

	/**
	 * Stores informations about the classification step.
	 */
	private ClassificationResult result;

	/**
	 * Constructor.
	 * 
	 * @param K
	 *            - number of nearest neighbors.
	 * @param distanceMeasure
	 *            - the distance function to calculate the similarity.
	 */
	public KNN(int K, DistanceMeasure distanceMeasure) {
		this.K = K;
		this.distanceMeasure = distanceMeasure;
		this.result = new ClassificationResult();
	}

	/**
	 * Read the ARFF file.
	 * 
	 * @param arffPath
	 *            - the path to the ARFF file
	 * @throws IOException
	 *             - if an error occurs when reading the file
	 */
	public ArffReader readArff(String arffPath) throws IOException {
		InputStream resourceAsStream = this.getClass().getResourceAsStream(arffPath);
		Reader reader = new InputStreamReader(resourceAsStream);
		return new ArffReader(reader);
	}

	/**
	 * Loads the ARFF file.
	 * 
	 * @param arffPath
	 *            - the path to the ARFF file
	 * @throws IOException
	 *             - if an error occurs when reading the file
	 */
	private void load(String arffPath) throws IOException {
		/* set the default value */
		if (arffPath == null) {
			arffPath = ARFF_FILE_PATH;
		}

		/* read the arff file */
		ArffReader reader = readArff(arffPath);
		this.instances = reader.getData();

		/* select the Class attribute */
		instances.setClassIndex(instances.numAttributes() - 1);
	}

	/**
	 * Run KNN.
	 * 
	 * @param arffPath
	 *            - the path to the ARFF file
	 * @throws IOException
	 *             - if an error occurs when reading the file
	 */
	public void run(String arffPath) throws IOException {
		/* load arff file */
		load(arffPath);

		/* stores the start time to calculate the total execution time */
		long startTime = System.nanoTime();

		/* do the math! */
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.instance(i);
			double[] attributes = instance.toDoubleArray();

			/* remove last item (it's the class) */
			attributes = Arrays.copyOf(attributes, attributes.length - 1);
			List<Pair<Double, Instance>> similarities = new ArrayList<Pair<Double, Instance>>();

			/* compare this instance with all other instances except itself */
			for (int j = 0; j < instances.numInstances(); j++) {
				if (i != j) {
					Instance otherInstance = instances.instance(j);
					double[] otherAttributes = otherInstance.toDoubleArray();

					/* remove last item (it's the class) */
					otherAttributes = Arrays.copyOf(otherAttributes, otherAttributes.length - 1);

					/* compute the distance */
					double similarity = distanceMeasure.compute(attributes, otherAttributes);
					Pair<Double, Instance> pair = new Pair<Double, Instance>(similarity, otherInstance);
					similarities.add(pair);
				}
			}

			/*
			 * the first indexes with the highest similarities (lowest
			 * distances)
			 */
			Collections.sort(similarities, new PairComparator());

			/* select the K nearest neighbors */
			List<Pair<Double, Instance>> neighbors = similarities.subList(0, this.K - 1);

			/* check the class with highest frequency */
			Double classz = highestFrequency(neighbors);
			if (classz == instance.value(instance.classIndex())) {
				this.result.incrementPositive();
			}
			this.result.incrementTotal();
		}

		/* calculate the total execution time */
		long stopTime = System.nanoTime();
		this.result.setExecutionTime((stopTime - startTime) / 1000000000f);
	}

	/**
	 * Given the K nearest neighbors, this method finds the class with highest
	 * frequency.
	 * 
	 * @param neighbors
	 *            - K nearest neighbors
	 * @return the class with highest frequency
	 */
	private Double highestFrequency(List<Pair<Double, Instance>> neighbors) {
		Map<Double, Integer> map = new HashMap<Double, Integer>();
		for (Pair<Double, Instance> i : neighbors) {
			Integer count = map.get(i.getKey());
			Instance second = i.getSecond();
			map.put(second.value(second.classIndex()), count != null ? count + 1 : 0);
		}

		Double popular = Collections.max(map.entrySet(), new Comparator<Map.Entry<Double, Integer>>() {
			public int compare(Entry<Double, Integer> o1, Entry<Double, Integer> o2) {
				return o1.getValue().compareTo(o2.getValue());
			}
		}).getKey();
		return popular;
	}

	/**
	 * Print results.
	 */
	public void printResult() {
		float correctly = result.getPositive();
		float incorrectly = result.getNumberOfPatterns() - correctly;

		float classificationRate = result.getPositive() / (float) result.getNumberOfPatterns();
		float errorRate = incorrectly / (float) result.getNumberOfPatterns();

		log.info("=== Dataset info ===");
		log.info("Dataset name = {}", this.instances.relationName());
		log.info("Number of instances = {}", this.instances.numInstances());
		log.info("Number of attributes = {}", this.instances.numAttributes());
		log.info("Number of classes = {}", this.instances.numClasses());

		log.info("=== Configuration ===");
		log.info("K = {}", this.K);
		log.info("Distance measure = {}", this.distanceMeasure.getClass().getSimpleName());

		log.info("=== Result ===");
		log.info("Correctly Classified Instances \t\t{} ({}%)", correctly, classificationRate * 100);
		log.info("Incorrectly Classified Instances \t\t{} ({}%)", incorrectly, errorRate * 100);
		log.info("Time taken to classify \t\t{} s", this.result.getExecutionTime());
	}
}
