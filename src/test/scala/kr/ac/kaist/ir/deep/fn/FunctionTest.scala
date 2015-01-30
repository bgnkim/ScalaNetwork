package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs
import org.specs2.mutable.Specification

/**
 * Test for functions
 */
class FunctionTest extends Specification {
  val input: ScalarMatrix = DenseMatrix.create(4, 1, Array[Double](0.0, 1.0, -0.35, 10.0))
  "Sigmoid" should {
    val out = Sigmoid(input)
    val diff = Sigmoid.derivative(out)

    "f(0) = 0.5" in {
      out(0, 0) must_=== 0.5
    }

    "f(1) = 0.73105857" in {
      out(1, 0) must beBetween(0.73105857, 0.73105858)
    }

    "f(-0.35) = 0.413382" in {
      out(2, 0) must beBetween(0.413382, 0.413383)
    }

    "f(10) = 0.999954" in {
      out(3, 0) must beBetween(0.999954, 0.999955)
    }

    "df/dx at y: 0.5 = 0.25" in {
      diff(0, 0) must_=== 0.25
    }

    "df/dx at y: 0.73105857 = 0.196612" in {
      diff(1, 1) must beBetween(0.196611, 0.196613)
    }

    "df/dx at y: 0.413382 = 0.242497" in {
      diff(2, 2) must beBetween(0.242496, 0.242498)
    }

    "df/dx at y: 0.999954 = 0.0000453958" in {
      diff(3, 3) must beBetween(0.0000453957, 0.0000453959)
    }
  }

  "Hyperbolic Tangent" should {
    val out = HyperbolicTangent(input)
    val diff = HyperbolicTangent.derivative(out)

    "f(0) = 0" in {
      out(0, 0) must_=== 0.0
    }
    "df/dx at y: 0 = 1" in {
      diff(0, 0) must_=== 1.0
    }

    "f(1) = 0.7615941559" in {
      out(1, 0) must beBetween(0.7615941559, 0.7615941560)
    }
    "df/dx at y: 0.7615941559 = 0.4199743416" in {
      diff(1, 1) must beBetween(0.4199743415, 0.4199743417)
    }

    "f(-0.35) = -0.336376" in {
      out(2, 0) must beBetween(-0.336377, -0.336375)
    }
    "df/dx at y: -0.336376 = 0.886851" in {
      diff(2, 2) must beBetween(0.886850, 0.886852)
    }

    "f(10) = 0.9999999958" in {
      out(3, 0) must beBetween(0.9999999957, 0.9999999959)
    }
    "df/dx at y: 0.9999999958 = 8.2446144e-9" in {
      diff(3, 3) must beBetween(8.244614e-9, 8.244615e-9)
    }
  }

  "Rectifier" should {
    val out = Rectifier(input)
    val diff = Rectifier.derivative(out)

    "f(0) = 0" in {
      out(0, 0) must_=== 0.0
    }
    "df/dx at y: 0 = 1" in {
      diff(0, 0) must_=== 0.0
    }

    "f(1) = 1" in {
      out(1, 0) must_=== 1
    }
    "df/dx at y: 1 = 1" in {
      diff(1, 1) must_=== 1
    }

    "f(-0.35) = 0" in {
      out(2, 0) must_=== 0
    }
    "df/dx at y: 0 = 0" in {
      diff(2, 2) must_=== 0
    }

    "f(10) = 10" in {
      out(3, 0) must_=== 10
    }
    "df/dx at y: 10 = 1" in {
      diff(3, 3) must_=== 1
    }
  }

  "Softplus" should {
    val out = Softplus(input)
    val diff = Softplus.derivative(out)

    "f(0) = 0.69314718" in {
      out(0, 0) must beBetween(0.69314717, 0.69314719)
    }
    "df/dx at y: 0.69314718 = 0.5" in {
      diff(0, 0) must beBetween(0.4999999, 0.5000001)
    }

    "f(1) = 1.313261687" in {
      out(1, 0) must beBetween(1.313261686, 1.313261688)
    }
    "df/dx at y: 1.313261687 = 0.73105857" in {
      diff(1, 1) must beBetween(0.73105857, 0.73105858)
    }

    "f(-0.35) = 0.533382155" in {
      out(2, 0) must beBetween(0.533382154, 0.533382156)
    }
    "df/dx at y: 0.533382155 = 0.413382" in {
      diff(2, 2) must beBetween(0.413382, 0.413383)
    }

    "f(10) = 10.0000453988" in {
      out(3, 0) must beBetween(10.0000453987, 10.0000453989)
    }
    "df/dx at y: 10.0000453988 = 0.999954" in {
      diff(3, 3) must beBetween(0.999954, 0.999955)
    }
  }

  "ScalarMatrix & its Op" should {
    "generate full-0 matrix" in {
      val matx = ScalarMatrix $0(10, 7)
      matx.rows must_=== 10
      matx.cols must_=== 7
      matx.toArray.toSet must containTheSameElementsAs(Seq(0))
    }

    "generate full-1 matrix" in {
      val matx = ScalarMatrix $1(11, 5)
      matx.rows must_=== 11
      matx.cols must_=== 5
      matx.toArray.toSet must containTheSameElementsAs(Seq(1))
    }

    "generate full-0/1 matrix" in {
      val matx = ScalarMatrix $01(3, 5, 0.5)
      matx.rows must_=== 3
      matx.cols must_=== 5
      matx.toArray.toSet must containTheSameElementsAs(Seq(0, 1))

      (0 until 100).foldLeft(0.0) {
        (f, _) ⇒
          val m = ScalarMatrix $01(3, 5, 0.5)
          f + m.toArray.toSeq.groupBy(x ⇒ x)(1.0).size
      } / 100 must beBetween(6.0, 8.0)
    }

    "generate full Random Matrix" in {
      val matx = ScalarMatrix of(5, 7)
      matx.rows must_=== 5
      matx.cols must_=== 7
      matx.toArray.toSet must have size be_>=(25)
    }

    "add 1 row with 7" in {
      val matx = ScalarMatrix of(5, 7)
      val newMatx = matx row_+ 7.0
      newMatx.rows must_=== 6
      newMatx.cols must_=== 7

      val row: ScalarMatrix = newMatx(5 to 5, ::)
      row.toArray.toSet must containTheSameElementsAs(Seq(7.0))
    }

    "concat two matrices" in {
      val a = ScalarMatrix $0(2, 3)
      val b = ScalarMatrix $1(2, 7)
      val ab = a col_+ b
      ab(0, 0) must_== 0.0
      ab(1, 7) must_== 1.0
      ab.cols must_== 10
      ab.rows must_== 2
    }

    "store & restore" in {
      val a = ScalarMatrix of(3, 5)
      val json = a.to2DSeq
      val b = ScalarMatrix restore json.as[Seq[Seq[Double]]]
      val d: Double = sum(abs(a - b))
      d must beBetween(0.0, 0.1)
    }
  }

  "ProbabilityOp" should {
    "safely retrieve value" in {
      1.0.safe must_== 1.0
      1.5.safe must_== 1.0
      0.5.safe must_== 0.5
      -1.0.safe must_== 0.0
      0.0.safe must_== 0.0
    }
  }


  "SquaredErr" should {
    val a = DenseMatrix.create(3, 1, Array(1.0, 0.0, 1.0))
    val b = DenseMatrix.create(3, 1, Array(0.0, 1.0, 1.0))
    val c = DenseMatrix.create(3, 1, Array(0.4, 0.2, 0.5))

    "do basic calculation" in {
      SquaredErr(a, b) must_== 1.0
      SquaredErr(a, c) must_== 0.325
    }

    "do diff calculation" in {
      val x = SquaredErr.derivative(a, b)
      x(0, 0) must_== -1.0
      x(0, 1) must_== 1.0
      x(0, 2) must_== 0.0
    }
  }

  "CrossEntropyErr" should {
    val a = DenseMatrix.create(3, 1, Array(1.0, 0.0, 1.0))
    val b = DenseMatrix.create(3, 1, Array(0.1, 0.9, 0.9))
    val c = DenseMatrix.create(3, 1, Array(0.4, 0.2, 0.5))

    "do basic calculation" in {
      CrossEntropyErr(a, b) must be_>=(0.0)
      CrossEntropyErr(a, c) must be_>=(0.0)
    }

    "do diff calculation" in {
      val x = CrossEntropyErr.derivative(a, b)
      x(0, 0) must beBetween(-10.0001, -9.999)
      x(0, 1) must beBetween(9.999, 10.0001)
      x(0, 2) must beBetween(-1.112, -1.110)
      val y = CrossEntropyErr.derivative(a, c)
      y(0, 0) must beBetween(-2.5001, -2.499)
      y(0, 1) must beBetween(1.2499, 1.25001)
      y(0, 2) must beBetween(-2.0001, -1.999)
    }
  }
}
