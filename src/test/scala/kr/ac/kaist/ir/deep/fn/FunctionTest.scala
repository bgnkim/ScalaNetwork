package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs
import org.specs2.mutable.Specification

/**
 * Test for functions
 */
class FunctionTest extends Specification {
  val input: ScalarMatrix = DenseMatrix.create(4, 1, Array[Scalar](0.0f, 1.0f, -0.35f, 10.0f))
  "Sigmoid" should {
    val out = Sigmoid(input)
    val diff = Sigmoid.derivative(out)

    "f(0) = 0.5" in {
      out(0, 0) must_=== 0.5f
    }

    "f(1) = 0.73105857" in {
      out(1, 0) must beBetween(0.73105857f, 0.73105858f)
    }

    "f(-0.35) = 0.413382" in {
      out(2, 0) must beBetween(0.413382f, 0.413383f)
    }

    "f(10) = 0.999954" in {
      out(3, 0) must beBetween(0.999954f, 0.999955f)
    }

    "df/dx at y: 0.5 = 0.25" in {
      diff(0, 0) must_=== 0.25f
    }

    "df/dx at y: 0.73105857 = 0.196612" in {
      diff(1, 1) must beBetween(0.196611f, 0.196613f)
    }

    "df/dx at y: 0.413382 = 0.242497" in {
      diff(2, 2) must beBetween(0.242496f, 0.242498f)
    }

    "df/dx at y: 0.999954 = 0.0000453958" in {
      diff(3, 3) must beBetween(0.0000453957f, 0.0000453959f)
    }
  }

  "Hyperbolic Tangent" should {
    val out = HyperbolicTangent(input)
    val diff = HyperbolicTangent.derivative(out)

    "f(0) = 0" in {
      out(0, 0) must_=== 0.0f
    }
    "df/dx at y: 0 = 1" in {
      diff(0, 0) must_=== 1.0f
    }

    "f(1) = 0.7615941559" in {
      out(1, 0) must beBetween(0.7615941559f, 0.7615941560f)
    }
    "df/dx at y: 0.7615941559 = 0.4199743416" in {
      diff(1, 1) must beBetween(0.4199743415f, 0.4199743417f)
    }

    "f(-0.35) = -0.336376" in {
      out(2, 0) must beBetween(-0.336377f, -0.336375f)
    }
    "df/dx at y: -0.336376 = 0.886851" in {
      diff(2, 2) must beBetween(0.886850f, 0.886852f)
    }

    "f(10) = 0.9999999958" in {
      out(3, 0) must beBetween(0.9999999957f, 0.9999999959f)
    }
    "df/dx at y: 0.9999999958 = 8.2446144e-9" in {
      diff(3, 3) must beBetween(8.244614e-9f, 8.244615e-9f)
    }
  }

  "Rectifier" should {
    val out = Rectifier(input)
    val diff = Rectifier.derivative(out)

    "f(0) = 0" in {
      out(0, 0) must_=== 0.0f
    }
    "df/dx at y: 0 = 1" in {
      diff(0, 0) must_=== 0.0f
    }

    "f(1) = 1" in {
      out(1, 0) must_=== 1f
    }
    "df/dx at y: 1 = 1" in {
      diff(1, 1) must_=== 1f
    }

    "f(-0.35) = 0" in {
      out(2, 0) must_=== 0f
    }
    "df/dx at y: 0 = 0" in {
      diff(2, 2) must_=== 0f
    }

    "f(10) = 10" in {
      out(3, 0) must_=== 10f
    }
    "df/dx at y: 10 = 1" in {
      diff(3, 3) must_=== 1f
    }
  }

  "Softplus" should {
    val out = Softplus(input)
    val diff = Softplus.derivative(out)

    "f(0) = 0.69314718" in {
      out(0, 0) must beBetween(0.69314717f, 0.69314719f)
    }
    "df/dx at y: 0.69314718 = 0.5" in {
      diff(0, 0) must beBetween(0.4999999f, 0.5000001f)
    }

    "f(1) = 1.313261687" in {
      out(1, 0) must beBetween(1.313261686f, 1.313261688f)
    }
    "df/dx at y: 1.313261687 = 0.73105857" in {
      diff(1, 1) must beBetween(0.73105857f, 0.73105858f)
    }

    "f(-0.35) = 0.533382155" in {
      out(2, 0) must beBetween(0.533382154f, 0.533382156f)
    }
    "df/dx at y: 0.533382155 = 0.413382" in {
      diff(2, 2) must beBetween(0.413382f, 0.413383f)
    }

    "f(10) = 10.0000453988" in {
      out(3, 0) must beBetween(10.0000453987f, 10.0000453989f)
    }
    "df/dx at y: 10.0000453988 = 0.999954" in {
      diff(3, 3) must beBetween(0.999954f, 0.999955f)
    }
  }

  "ScalarMatrix & its Op" should {
    "generate full-0 matrix" in {
      val matx = ScalarMatrix $0(10, 7)
      matx.rows must_=== 10
      matx.cols must_=== 7
      matx.toArray.toSet must containTheSameElementsAs(Seq(0f))
    }

    "generate full-1 matrix" in {
      val matx = ScalarMatrix $1(11, 5)
      matx.rows must_=== 11
      matx.cols must_=== 5
      matx.toArray.toSet must containTheSameElementsAs(Seq(1f))
    }

    "generate full-0/1 matrix" in {
      val matx = ScalarMatrix $01(3, 5, 0.5f)
      matx.rows must_=== 3
      matx.cols must_=== 5
      matx.toArray.toSet must containTheSameElementsAs(Seq(0, 1))

      (0 until 100).foldLeft(0.0f) {
        (f, _) ⇒
          val m = ScalarMatrix $01(3, 5, 0.5f)
          f + m.toArray.toSeq.groupBy(x ⇒ x)(1.0f).size
      } / 100 must beBetween(6.0f, 8.0f)
    }

    "generate full Random Matrix" in {
      val matx = ScalarMatrix of(5, 7)
      matx.rows must_=== 5
      matx.cols must_=== 7
      matx.toArray.toSet must have size be_>=(25)
    }

    "add 1 row with 7" in {
      val matx = ScalarMatrix of(5, 7)
      val newMatx = matx row_+ 7.0f
      newMatx.rows must_=== 6
      newMatx.cols must_=== 7

      val row: ScalarMatrix = newMatx(5 to 5, ::)
      row.toArray.toSet must containTheSameElementsAs(Seq(7.0f))
    }

    "concat two matrices" in {
      val a = ScalarMatrix $0(2, 3)
      val b = ScalarMatrix $1(2, 7)
      val ab = a col_+ b
      ab(0, 0) must_== 0.0f
      ab(1, 7) must_== 1.0f
      ab.cols must_== 10
      ab.rows must_== 2
    }

    "store & restore" in {
      val a = ScalarMatrix of(3, 5)
      val json = a.to2DSeq
      val b = ScalarMatrix restore json.as[IndexedSeq[IndexedSeq[Float]]]
      val d = sum(abs(a - b))
      d must beBetween(0.0f, 0.1f)
    }
  }

  "ProbabilityOp" should {
    "safely retrieve value" in {
      1.0f.safe must_== 1.0f
      1.5f.safe must_== 1.0f
      0.5f.safe must_== 0.5f
      -1.0f.safe must_== 0.0f
      0.0f.safe must_== 0.0f
    }
  }


  "SquaredErr" should {
    val a = DenseMatrix.create(3, 1, Array(1.0f, 0.0f, 1.0f))
    val b = DenseMatrix.create(3, 1, Array(0.0f, 1.0f, 1.0f))
    val c = DenseMatrix.create(3, 1, Array(0.4f, 0.2f, 0.5f))

    "do basic calculation" in {
      SquaredErr(a, b) must_== 2.0f
      SquaredErr(a, c) must_== 0.65f
    }

    "do diff calculation" in {
      val x = SquaredErr.derivative(a, b)
      x(0, 0) must_== -1.0f
      x(1, 0) must_== 1.0f
      x(2, 0) must_== 0.0f
    }
  }

  "CrossEntropyErr" should {
    val a = DenseMatrix.create(3, 1, Array(1.0f, 0.0f, 1.0f))
    val b = DenseMatrix.create(3, 1, Array(0.1f, 0.9f, 0.9f))
    val c = DenseMatrix.create(3, 1, Array(0.4f, 0.2f, 0.5f))

    "do basic calculation" in {
      CrossEntropyErr(a, b) must be_>=(0.0f)
      CrossEntropyErr(a, c) must be_>=(0.0f)
    }

    "do diff calculation" in {
      val x = CrossEntropyErr.derivative(a, b)
      x(0, 0) must beBetween(-10.0001f, -9.999f)
      x(1, 0) must beBetween(9.999f, 10.0001f)
      x(2, 0) must beBetween(-1.112f, -1.110f)
      val y = CrossEntropyErr.derivative(a, c)
      y(0, 0) must beBetween(-2.5001f, -2.499f)
      y(1, 0) must beBetween(1.2499f, 1.25001f)
      y(2, 0) must beBetween(-2.0001f, -1.999f)
    }
  }
}
