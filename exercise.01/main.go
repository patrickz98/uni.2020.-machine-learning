package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/patrickz98/uni.2020.machine-learning/simple"
)

const (
	notebookPath = "exercise.01.notebook"
	iterations   = 5000
)

type SimpleExport struct {
	XPoints []float64
	YPoints []float64
}

type SGDExport struct {
	Name             string
	Thetas           []float64
	FunctionStr      string
	Iterations       int
	PolynomialDegree int
	LearnRate        float64
	XPoints          []float64
	YPoints          []float64
	Error            []float64
}

type Point struct {
	X     float64
	Y     float64
	Noise float64
}

type Points []Point

func (points Points) getXY() (xs []float64, ys []float64) {

	xs = make([]float64, len(points))
	ys = make([]float64, len(points))

	for inx, point := range points {
		xs[inx] = point.X
		ys[inx] = point.Y
	}

	return xs, ys
}

func (points Points) export() SimpleExport {

	xi, yi := points.getXY()
	return SimpleExport{
		XPoints: xi,
		YPoints: yi,
	}
}

func randFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

// (1) Generate 100 artificial data points (xi,yi) where
// each xi is randomly generated from the interval [0, 1]
// and yi = sin(2πxi) + ε. Here, ε is a random noise
// value in the interval [−0.3, 0.3].
func generateRandomPoints(num int) Points {

	points := make(Points, num)
	stepSize := 1.0 / float64(num)

	for inx := 0; inx < num; inx++ {

		noise := randFloat(-0.3, 0.3)

		x := stepSize * float64(inx)
		y := math.Sin(2*math.Pi*x) + noise

		point := Point{
			X:     x,
			Y:     y,
			Noise: noise,
		}

		points[inx] = point
	}

	return points
}

// convert thetas to a function string
func thetas2FunctionString(thetas []float64) string {

	parts := make([]string, len(thetas))
	for inx, th := range thetas {
		parts[inx] = fmt.Sprintf("%f * x ^ %d", th, inx)
	}

	return "y = " + strings.Join(parts, " + ")
}

// get points for h in interval form 0.0 to 1.0 for step size.
func plotFunction(steps int, thetas []float64) Points {

	step := 1.0 / float64(steps)
	points := make(Points, steps)

	for inx := 0; inx < steps; inx++ {

		x := step * float64(inx)

		points[inx] = Point{
			X: x,
			Y: hypotheses(x, thetas),
		}
	}

	return points
}

// h theta function for x
func hypotheses(x float64, thetas []float64) float64 {

	sum := 0.0

	for idx := 0; idx < len(thetas); idx++ {
		sum += thetas[idx] * math.Pow(x, float64(idx))
	}

	return sum
}

// Stochastic Gradient Descent algorithm
func stochasticGradientDescent(
	// training data with x and y
	trainingPoints Points,
	// learn rate aka alpher
	learnRate float64,
	// degree of the polynomial (D)
	polynomialDegree int) SGDExport {

	thetas := make([]float64, polynomialDegree+1)
	for inx := 0; inx <= polynomialDegree; inx++ {
		thetas[inx] = randFloat(-0.5, 0.5)
	}

	errorRate := make([]float64, iterations)
	for idx := 0; idx < iterations; idx++ {

		for _, point := range trainingPoints {
			for j := range thetas {

				// The last factor (xi)j means:
				// the factor multiplying parameterθj in
				// the polynmial function, which in this
				// case it will be xi to the power of j.
				xij := math.Pow(point.X, float64(j))

				thetas[j] += learnRate * (point.Y - hypotheses(point.X, thetas)) * xij
			}
		}

		jerror := 0.0

		for _, point := range trainingPoints {
			jerror += math.Pow(hypotheses(point.X, thetas)-point.Y, float64(2))
		}

		errorRate[idx] = jerror * 0.5
	}

	plot := plotFunction(len(trainingPoints), thetas)
	xs, ys := plot.getXY()

	return SGDExport{
		Name:             fmt.Sprintf("D=%v a=%v", polynomialDegree, learnRate),
		Thetas:           thetas,
		FunctionStr:      thetas2FunctionString(thetas),
		Iterations:       iterations,
		PolynomialDegree: polynomialDegree,
		LearnRate:        learnRate,
		XPoints:          xs,
		YPoints:          ys,
		Error:            errorRate,
	}
}

func main() {

	//
	// Setup
	//

	// seed with birthday to get same results for every try.
	rand.Seed(28051998)

	// mkdir data export dir
	_ = os.MkdirAll(notebookPath, 0755)

	//
	// Stochastic Gradient Discent
	//

	// generate 100 random points.
	randomPoints := generateRandomPoints(100)

	// export and save generated points
	plot := randomPoints.export()
	simple.WritePretty(plot, notebookPath+"/exercise.01.json")

	examples := []SGDExport{
		// stochasticGradientDescent(randomPoints, 0.001, 3),
		// stochasticGradientDescent(randomPoints, 0.001, 4),
		// stochasticGradientDescent(randomPoints, 0.001, 5),
		// stochasticGradientDescent(randomPoints, 0.001, 6),
		// stochasticGradientDescent(randomPoints, 0.01, 3),
		// stochasticGradientDescent(randomPoints, 0.01, 4),
		// stochasticGradientDescent(randomPoints, 0.01, 5),
		// stochasticGradientDescent(randomPoints, 0.01, 6),
		stochasticGradientDescent(randomPoints, 0.01, 7),
	}

	simple.WritePretty(examples, notebookPath+"/exercise.01.sgd.json")
}
