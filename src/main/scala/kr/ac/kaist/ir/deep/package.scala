package kr.ac.kaist.ir

/**
 * A ''Neural Network implementation'' with Scala, [[https://github.com/scalanlp/breeze Breeze]] & [[http://spark.apache.org Spark]]
 *
 * @example
 * {{{// Define 2 -> 4 -> 1 Layered, Fully connected network.
 *   val net = Network(Sigmoid, 2, 4, 1)
 *
 *    // Define Manipulation Type. VectorType, AEType, RAEType, and URAEType.
 *    val operation = new VectorType(
 *       corrupt = GaussianCorruption(variance = 0.1)
 *    )
 *
 *   // Define Training Style. SingleThreadTrainStyle vs DistBeliefTrainStyle
 *    val style = new SingleThreadTrainStyle(
 *      net = net,
 *      algorithm = new StochasticGradientDescent(l2decay = 0.0001),
 *       make = operation,
 *      param = SimpleTrainingCriteria(miniBatch = 8))
 *
 *   // Define Trainer
 *   val train = new Trainer(
 *      style = style,
 *      stops = StoppingCriteria(maxIter = 100000))
 *
 *   // Do Train
 *   train.train(set, valid)}}}
 */
package object deep
