package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network, NeuralObject}

import scala.annotation.tailrec

/**
 * Package for training
 *
 * Created by bydelta on 2015-01-02.
 */
package object trainer {
  /** Type of Corruption */
  type Corruption = NeuronVector ⇒ NeuronVector

  /**
   * Abstract class for weight update
   */
  abstract class WeightUpdater extends (Seq[NeuralObject] ⇒ Unit) with Serializable {
    /** Decay factor for L1-reg */
    val l1decay: Double
    /** Decay factor for L2-reg */
    val l2decay: Double

    /**
     * Compute weight-loss of given neuron objects
     * @param objs to be computed
     * @return weight loss of the set
     */
    def loss(objs: Seq[NeuralObject]) = {
      val l1loss = objs.L1 * l1decay
      val l2loss = objs.L2 * l2decay
      l1loss + l2loss
    }
  }

  /**
   * Trait: Trainer
   */
  trait Trainer extends Serializable {
    /** Stopping Criteria */
    protected[trainer] val stops: StoppingCriteria

    /**
     * Train given sequence, and validate with given sequence.
     * @param set to be trained
     * @return Training error (loss)
     */
    def train(set: Seq[(NeuronVector, NeuronVector)]): Scalar = trainWithValidation(set, set)

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training
     * @param valid to be used for validation
     * @return Training error (loss)
     */
    def trainWithValidation(set: Seq[(NeuronVector, NeuronVector)], valid: Seq[(NeuronVector, NeuronVector)]): Scalar
  }

  /**
   * Input Corruption: Drop input as zero
   * @param dropRate probability
   */
  case class DroppingCorruption(dropRate: Double = 0.01) extends Corruption {
    /**
     * Do corruption
     * @param v1 to be corrupted
     * @return corrupted vector
     */
    override def apply(v1: NeuronVector): NeuronVector = v1 mapValues { x ⇒ if (UniformNoise() > dropRate) x else 0.0}
  }

  /**
   * Input Corruption: Gaussian
   * @param mean of noise
   * @param stdev of noise
   */
  case class GaussianCorruption(mean: Double = 0.0, stdev: Double = 1.0) extends Corruption {
    /**
     * Do corruption
     * @param v1 to be corrupted
     * @return corrupted vector
     */
    override def apply(v1: NeuronVector): NeuronVector = v1 mapValues { x ⇒ x + GaussianNoise(mean, stdev)}
  }

  /**
   * Input Corruption: None
   */
  case object NoCorruption extends Corruption {

    /**
     * Identity.
     * @param v1 to be corrupted
     * @return the vector
     */
    override def apply(v1: NeuronVector) = v1
  }

  /**
   * Criteria: When to stop training
   * @param maxIter is maximum iteration count
   * @param patience is default patience count
   * @param patienceStep is default step for patience
   * @param improveThreshold is threshold for marked as "improved"
   * @param validationFreq is step count for validation
   */
  case class StoppingCriteria(maxIter: Int = 10000, patience: Int = 5000, patienceStep: Int = 2, improveThreshold: Double = 0.995, validationFreq: Int = 10)

  /**
   * Criteria: How to train
   * @param batch is size of mini-batch
   * @param dropout is weight drop-out probability
   */
  case class TrainingCriteria(batch: Int = 10, dropout: Double = 0.001)

  /**
   * Operation: Neural Objects
   * @param seq to be computed
   */
  implicit class NeuralObjectOp(seq: Seq[NeuralObject]) {
    /**
     * Weight vector
     * @return weight of neural objects
     */
    def w = seq map { n ⇒ Math.abs(n.weight.value)}

    /**
     * L1-norm
     * @return size of weight vector
     */
    def L1 = w.sum

    /**
     * Squared L2-norm
     * @return size of weight vector
     */
    def L2 = (w map { x ⇒ x * x}).sum
  }

  /**
   * Algorithm: Gradient Descent
   * @param rate for learning
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param momentum for momentum adaptive learning
   */
  class GradientDescent(rate: Double = 0.6, override val l1decay: Double = 0.01, override val l2decay: Double = 0.01, momentum: Double = 0.001) extends WeightUpdater {
    /** Last update */
    var lastDelta = Map[Long, Scalar]()

    /**
     * Run the algorithm
     * @param seq to be applied
     */
    override def apply(seq: Seq[NeuralObject]): Unit = {
      lastDelta = (seq map {
        o ⇒ {
          val w = o.weight.value

          val deltaW = o.weight.delta
          val deltal1 = if (w > 0) l1decay else if (w < 0) -l1decay else 0.0
          val deltal2 = w * l2decay * 2
          val moment = lastDelta.getOrElse(o.weight.id, 0.0) * momentum
          val dw = moment - (deltaW + deltal1 + deltal2) * rate

          o.weight += dw
          o.weight.id → dw
        }
      }).toMap
    }
  }

  /**
   * Algorithm: AdaGrad
   * @param rate for learning
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param historyCount is size of last update history
   */
  class AdaGrad(rate: Double = 0.6, override val l1decay: Double = 0.01, override val l2decay: Double = 0.01, val historyCount: Int = 10) extends WeightUpdater {
    /** History of update */
    var deltaSeq = Seq[Scalar]()

    /**
     * Run the algorithm
     * @param seq to be updated
     */
    override def apply(seq: Seq[NeuralObject]): Unit = {
      val grad = (seq map {
        o ⇒ {
          val w = o.weight.value

          val deltaW = o.weight.delta
          val deltal1 = if (w > 0) l1decay else if (w < 0) -l1decay else 0.0
          val deltal2 = w * l2decay * 2
          o.weight.id → (deltaW + deltal1 + deltal2)
        }
      }).toMap

      val gradL2 = grad.foldLeft(0.0) { (sum, pair) ⇒ sum + pair._2 * pair._2}
      deltaSeq = gradL2 +: deltaSeq.take(historyCount)

      val adaptedRate = rate / (if (deltaSeq.isEmpty) 1.0 else Math.sqrt(deltaSeq.sum))

      seq map {
        o ⇒ {
          val dw = grad(o.weight.id) * adaptedRate

          o.weight += dw
        }
      }
    }
  }

  /**
   * Algorithm: AdaDelta
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param decay for AdaDelta history decay factor
   * @param epsilon for AdaDelta base factor
   */
  class AdaDelta(override val l1decay: Double = 0.01,
                 override val l2decay: Double = 0.01,
                 val decay: Double = 0.95,
                 val epsilon: Double = 1e-6)
    extends WeightUpdater {
    /** History of Delta */
    var avgDeltaL2 = 0.0
    /** History of Gradient */
    var avgGradL2 = 0.0

    /**
     * Run the algorithm
     * @param seq to be applied
     */
    override def apply(seq: Seq[NeuralObject]): Unit = {
      val grad = (seq map {
        o ⇒ {
          val w = o.weight.value

          val deltaW = o.weight.delta
          val deltal1 = if (w > 0) l1decay else if (w < 0) -l1decay else 0.0
          val deltal2 = w * l2decay * 2
          o.weight.id → (deltaW + deltal1 + deltal2)
        }
      }).toMap

      val gradL2 = grad.foldLeft(0.0) { (sum, pair) ⇒ sum + pair._2 * pair._2} / seq.size
      avgGradL2 = decay * avgGradL2 + (1.0 - decay) * gradL2

      val rate = Math.sqrt(avgDeltaL2 + epsilon) / Math.sqrt(avgGradL2 + epsilon)

      val delta = seq.foldLeft(0.0) {
        (sum, o) ⇒ {
          val dw = grad(o.weight.id) * rate

          o.weight += dw
          sum + dw * dw
        }
      } / seq.size

      avgDeltaL2 = decay * avgDeltaL2 + (1.0 - decay) * delta
    }
  }

  /**
   * Trainer : Stochastic-Style
   * @param net to be trained
   * @param algorithm to be applied
   * @param error to compute loss
   * @param corrupt to corrupt input
   * @param param of training criteria
   * @param stops of stopping criteria
   */
  class StochasticTrainer(val net: Network,
                          val algorithm: WeightUpdater,
                          val error: Objective = SquaredErr,
                          val corrupt: Corruption = NoCorruption,
                          val param: TrainingCriteria = TrainingCriteria(),
                          override val stops: StoppingCriteria = StoppingCriteria())
    extends Trainer {
    /** Stochastic Generator Base */
    protected[trainer] val randomSetGenerator = (set: Seq[(NeuronVector, NeuronVector)]) ⇒ () ⇒ set((Math.random() * set.size).toInt)
    /** One-time trainer of network */
    protected[trainer] val trainer = net.trainerOf(error) _
    /** Generator */
    protected[trainer] var generate: () ⇒ (NeuronVector, NeuronVector) = null
    /** Validation Set */
    protected[trainer] var validation: Seq[(NeuronVector, NeuronVector)] = null
    /** Best Parameter History */
    protected[trainer] var bestParam: Map[Long, Scalar] = null

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training
     * @param valid to be used for validation
     * @return Training error (loss)
     */
    def trainWithValidation(set: Seq[(NeuronVector, NeuronVector)], valid: Seq[(NeuronVector, NeuronVector)]) = {
      generate = randomSetGenerator(set)
      validation = valid

      val err = trainBatch()
      restoreParams()
      net.synapses foreach (_.dropIf(clause = false))
      err
    }

    /**
     * Store best parameters
     */
    protected[trainer] final def saveParams() = {
      bestParam = (net.all map { n ⇒ n.weight.id → n.weight.value}).toMap
    }

    /**
     * Restore best parameters
     */
    protected[trainer] final def restoreParams() = {
      net.all foreach { n ⇒ n.weight := bestParam(n.weight.id)}
    }

    /**
     * Tail Recursive : Train each batch
     * @param iter indicates current iteration
     * @param prevloss indicates previous loss
     * @param patience indicates current patience
     * @return Total Loss when train is finished
     */
    @tailrec
    protected[trainer] final def trainBatch(iter: Int = 0, prevloss: Double = Double.MaxValue, patience: Int = stops.patience): Scalar = {
      (0 until param.batch) foreach { _ ⇒ {
        val pair = generate()
        val in = corrupt(pair._1)
        net.synapses foreach (_.dropIf(Math.random() < param.dropout))
        trainer(in, pair._2)
      }
      }
      net ! algorithm

      var nPatience = patience

      val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
        net.synapses foreach (_.dropIf(clause = false))
        val train = validation.foldLeft(0.0) { (err, item) ⇒ err + error(item._2, net(item._1))} / validation.size
        val weight = algorithm loss net.all
        if (train + weight < prevloss * stops.improveThreshold) {
          nPatience = Math.max(patience, iter * stops.patienceStep)
          saveParams()
          println(f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f")
          train + weight
        } else {
          prevloss
        }
      } else {
        prevloss
      }

      if (iter < param.batch * stops.maxIter && nPatience > iter) {
        trainBatch(iter + 1, nLoss, nPatience)
      } else {
        println(f"Finished $iter%6d, Error = $nLoss%.5f")
        nLoss
      }
    }
  }

  /**
   * Trainer : Auto-Encoder trainer based on Stochastic-style
   * @param net to be trained
   * @param algorithm to be applied
   * @param objective to compute loss
   * @param corrupt to corrupt input
   * @param param of training criteria
   * @param stops of stopping criteria
   */
  class EncoderTrainer(override val net: AutoEncoder,
                       override val algorithm: WeightUpdater = new GradientDescent(),
                       val objective: Objective = SquaredErr,
                       override val corrupt: Corruption = NoCorruption,
                       override val param: TrainingCriteria = TrainingCriteria(),
                       override val stops: StoppingCriteria = StoppingCriteria())
    extends StochasticTrainer(net, algorithm, objective, corrupt, param, stops) {
    /** Ouput Index for conversion */
    private val outIndexes = net.start + net.input + net.hidden
    /** New Random Generator for AutoEncoder */
    protected[trainer] val randomGenerator = (set: Seq[NeuronVector]) ⇒ () ⇒ {
      val x = set((Math.random() * set.size).toInt)
      val y = x map { pair ⇒ (outIndexes + pair._1) → pair._2}
      x → y
    }

    /**
     * Train given sequence, and validate with that sequence.
     * @param set to be used for training
     * @return Trained Loss
     */
    def trainAuto(set: Seq[NeuronVector]) = {
      generate = randomGenerator(set)
      validation = set map { x ⇒ x → x}

      val err = trainBatch()
      restoreParams()
      net.synapses foreach (_.dropIf(clause = false))
      err
    }
  }

}
