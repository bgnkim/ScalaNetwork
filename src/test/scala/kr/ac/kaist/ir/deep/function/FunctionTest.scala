package kr.ac.kaist.ir.deep.function

import org.specs2.mutable.Specification

/**
 * Created by bydelta on 2015-01-03.
 */
class FunctionTest extends Specification {
  "Sigmoid" should {
    "f(0) = 0.5" in {
      Sigmoid(0.0) must_=== 0.5
    }

    "f(1) = 0.73105857" in {
      Sigmoid(1) must beBetween(0.73105857, 0.73105858)
    }

    "f(-0.35) = 0.413382" in {
      Sigmoid(-0.35) must beBetween(0.413382, 0.413383)
    }

    "f(10) = 0.999954" in {
      Sigmoid(10) must beBetween(0.999954, 0.999955)
    }

    "df/dx at y: 0.5 = 0.25" in {
      Sigmoid.derivative(0.5) must_=== 0.25
    }

    "df/dx at y: 0.73105857 = 0.196612" in {
      Sigmoid.derivative(Sigmoid(1)) must beBetween(0.196611, 0.196613)
    }

    "df/dx at y: 0.413382 = 0.242497" in {
      Sigmoid.derivative(Sigmoid(-0.35)) must beBetween(0.242496, 0.242498)
    }

    "df/dx at y: 0.999954 = 0.0000453958" in {
      Sigmoid.derivative(Sigmoid(10)) must beBetween(0.0000453957, 0.0000453959)
    }
  }

  "Hyperbolic Tangent" should {
    "f(0) = 0" in {
      HyperbolicTangent(0.0) must_=== 0.0
    }
    "df/dx at y: 0 = 1" in {
      HyperbolicTangent.derivative(0.0) must_=== 1.0
    }

    "f(1) = 0.7615941559" in {
      HyperbolicTangent(1) must beBetween(0.7615941559, 0.7615941560)
    }
    "df/dx at y: 0.7615941559 = 0.4199743416" in {
      HyperbolicTangent.derivative(HyperbolicTangent(1)) must beBetween(0.4199743415, 0.4199743417)
    }

    "f(-0.35) = -0.336376" in {
      HyperbolicTangent(-0.35) must beBetween(-0.336377, -0.336375)
    }
    "df/dx at y: -0.336376 = 0.886851" in {
      HyperbolicTangent.derivative(HyperbolicTangent(-0.35)) must beBetween(0.886850, 0.886852)
    }

    "f(10) = 0.9999999958" in {
      HyperbolicTangent(10) must beBetween(0.9999999957, 0.9999999959)
    }
    "df/dx at y: 0.9999999958 = 8.2446144e-9" in {
      HyperbolicTangent.derivative(HyperbolicTangent(10)) must beBetween(8.244614e-9, 8.244615e-9)
    }
  }

  "Rectifier" should {

    "f(0) = 0" in {
      Rectifier(0.0) must_=== 0.0
    }
    "df/dx at y: 0 = 1" in {
      Rectifier.derivative(0.0) must_=== 0.0
    }

    "f(1) = 1" in {
      Rectifier(1) must_=== 1
    }
    "df/dx at y: 1 = 1" in {
      Rectifier.derivative(Rectifier(1)) must_=== 1
    }

    "f(-0.35) = 0" in {
      Rectifier(-0.35) must_=== 0
    }
    "df/dx at y: 0 = 0" in {
      Rectifier.derivative(Rectifier(-0.35)) must_=== 0
    }

    "f(10) = 10" in {
      Rectifier(10) must_=== 10
    }
    "df/dx at y: 10 = 1" in {
      Rectifier.derivative(Rectifier(10)) must_=== 1
    }
  }

  "Softplus" should {

    "f(0) = 0.69314718" in {
      Softplus(0.0) must beBetween(0.69314717, 0.69314719)
    }
    "df/dx at y: 0.69314718 = 0.5" in {
      Softplus.derivative(Softplus(0.0)) must beBetween(0.4999999, 0.5000001)
    }

    "f(1) = 1.313261687" in {
      Softplus(1) must beBetween(1.313261686, 1.313261688)
    }
    "df/dx at y: 1.313261687 = 0.73105857" in {
      Softplus.derivative(Softplus(1)) must beBetween(0.73105857, 0.73105858)
    }

    "f(-0.35) = 0.533382155" in {
      Softplus(-0.35) must beBetween(0.533382154, 0.533382156)
    }
    "df/dx at y: 0.533382155 = 0.413382" in {
      Softplus.derivative(Softplus(-0.35)) must beBetween(0.413382, 0.413383)
    }

    "f(10) = 10.0000453988" in {
      Softplus(10) must beBetween(10.0000453987, 10.0000453989)
    }
    "df/dx at y: 10.0000453988 = 0.999954" in {
      Softplus.derivative(Softplus(10)) must beBetween(0.999954, 0.999955)
    }
  }

  "Gaussian Noise" should {
    "Generate mean 0 stdev 1 Gaussian" in {
      val set = (0 until 100000) map { _ ⇒ GaussianNoise()}
      set.sum / 100000 must beBetween(-0.01, 0.01)
      (set map { x ⇒ x * x}).sum / 100000 must beBetween(0.99, 1.01)
    }
  }

  "NeuronVector Operations" should {
    val a = Map(1 → 0.0, 2 → 1.0, 3 → 0.0, 4 → 1.0)
    val b = Map(1 → 1.0, 2 → -1.0, 4 → 0.0, 5 → 1.0)
    "Add properly" in {
      a + b must havePairs(1 → 1.0, 2 → 0.0, 3 → 0.0, 4 → 1.0, 5 → 1.0)
    }

    "Subtract properly" in {
      a - b must havePairs(1 → -1.0, 2 → 2.0, 3 → 0.0, 4 → 1.0, 5 → -1.0)
    }

    "Elementwise product properly" in {
      a * b must havePairs(1 → 0.0, 2 → -1.0, 3 → 0.0, 4 → 0.0, 5 → 0.0)
    }

    "Scala product properly" in {
      a * 2 must havePairs(1 → 0.0, 2 → 2.0, 3 → 0.0, 4 → 2.0)
    }

    "Dot product properly" in {
      a dot b must_=== -1.0
    }
  }
}
