package kr.ac.kaist.ir.deep.train

import org.apache.spark.Partitioner

/**
 * Spark Partitioner that gives almost-equal partitions.
 *
 * @note Use this with RDD.zipWithUniqueId()
 *
 * @param numPartition Number of partitions
 */
class RandomEqualPartitioner(val numPartition: Int) extends Partitioner {
  private var nextNumber = 0

  def refreshRandom() = {
    nextNumber += 1
  }

  override def numPartitions: Int = numPartition

  override def getPartition(key: Any): Int = {
    val i = key.asInstanceOf[Long] + nextNumber
    val remain = i % numPartition
    val quotient = ((i / numPartition) * nextNumber) % numPartition
    val hash = ((remain + quotient) % numPartition).asInstanceOf[Int]
    if (hash < 0)
      hash + numPartition
    else
      hash
  }
}
