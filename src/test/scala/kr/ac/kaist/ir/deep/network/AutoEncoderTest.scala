package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.trainer._
import org.specs2.mutable.Specification
import play.api.libs.json.JsObject

/**
 * Created by bydelta on 2015-01-03.
 */
class AutoEncoderTest extends Specification {
  val structureTest = new AutoEncoder(3, 2, Sigmoid)
  "AutoEncoder" should {
    "have right size" in {
      structureTest.neurons must haveLength(2)
      structureTest.synapses must haveLength(12)
      structureTest.outNeurons must haveLength(3)
    }

    "have right numbering" in {
      val nid = structureTest.neurons map { n ⇒ n.id}
      nid.min must_=== 4
      nid.max must_=== 5

      val oid = structureTest.outNeurons map { n ⇒ n.id}
      oid.min must_=== 6
      oid.max must_=== 8

      val pairs = structureTest.synapses map { s ⇒ s.src1 → s.dst}
      pairs must havePairs(1 → 4, 2 → 4, 3 → 4, 1 → 5, 2 → 5, 3 → 5, 4 → 6, 4 → 7, 4 → 8, 5 → 6, 5 → 7, 5 → 8)
    }

    "properly serializable & reloadable" in {
      val obj = structureTest.toJSON
      obj must haveClass[JsObject]
      val newNet = Network.fromJSON(obj)
      newNet.neurons must haveLength(2)
      newNet.synapses must haveLength(12)
      newNet.outNeurons must haveLength(3)
    }
  }

  val encoder = new AutoEncoder(6, 3, Sigmoid)
  val sgdTrainer = new EncoderTrainer(encoder, algorithm = new AdaGrad(), corrupt = GaussianCorruption(stdev = 0.02))
  "EncoderTrainer" should {
    var trainset = List[NeuronVector]()
    for (i1 ← 0 to 1; i2 ← 0 to 1; i3 ← 0 to 1) {
      val x = Map(1 → i1.toDouble, 2 → i2.toDouble, 3 → i3.toDouble, 4 → i1.toDouble, 5 → i2.toDouble, 6 → i3.toDouble)
      trainset = x :: trainset
    }

    "generate proper trainset" in {
      trainset must haveLength(8)
    }
    "properly trained" in {
      sgdTrainer.trainAuto(trainset) must be_<(1.0)
    }

    examplesBlock {
      for (i ← 0 until 8) {
        val item = trainset(i)
        val out = encoder(item) map { pair ⇒ (pair._1 - 9) → pair._2}
        f"example $i : $item -> $out" in {
          SquaredErr(item, out) must be_<(0.12)
        }
      }
    }
  }
}
