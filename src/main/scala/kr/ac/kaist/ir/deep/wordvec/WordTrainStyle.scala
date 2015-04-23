package kr.ac.kaist.ir.deep.wordvec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train._
import org.apache.spark.SparkContext

import scala.collection.mutable

/**
 * __Trainer__ : Stochastic-Style, Multi-Threaded WordEmbedding Train Style using Spark.
 *
 * @param net __Network__ to be trained
 * @param algorithm Weight __update algorithm__ to be applied
 * @param make __String-String Input Operation__ that supervises how to manipulate input as matrices.
 *             This also controls how to compute actual network.
 * @param param __Training criteria__ (default: [[kr.ac.kaist.ir.deep.train.DistBeliefCriteria]])
 */
class WordTrainStyle(net: Network,
                     algorithm: WeightUpdater,
                     @transient sc: SparkContext,
                     make: StringToStringType,
                     objective: Objective = SquaredErr,
                     param: DistBeliefCriteria = DistBeliefCriteria())
  extends MultiThreadTrainStyle[String, String](net, algorithm, sc, make, param) {

  /** Set Accumulator variable for networks */
  protected val accWord = sc.accumulator(mutable.HashMap[String, ScalarMatrix]())
  make.setAccumulator(accWord)

  //TODO fetch and update, save param....
}