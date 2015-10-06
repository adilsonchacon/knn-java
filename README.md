# knn-java
A KNN implemention in Java.

## Build in Eclipse
1. Clone the project.
2. Using eclipse, select `File > Import`.Under `Maven`, choose `Existing Maven Projects`.
3. Point to the directory where the `pom.xml` is stored.
4. Select the project and then `Finish`.
5. Install dependencies and build with `Right click on the project > Run as > Maven install`.

## Usage
Using the project is very straightforward:
```java
try {
  KNN knn = new KNN(5, new EuclideanDistance());
  knn.run(KNN.ARFF_FILE_PATH);
  knn.printResult();
} catch (IOException e) {
  // file issues
}
```		

## Options
We use Apache's [DistanceMeasure][1] interface. Therefore, we support any implementation, including these out-of-the-box:

Distance function  | Call
------------- | -------------
Euclidean  | new EuclideanDistance()
Canberra  | new CanberraDistance()
Chebyshev  | new ChebyshevDistance()
Earth mover's  | new EarthMoversDistance()
Manhattan  | new ManhattanDistance()

[1]: https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math3/ml/distance/package-summary.html
