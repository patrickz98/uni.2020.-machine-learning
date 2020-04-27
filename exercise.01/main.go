package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/patrickz98/uni.2020.machine-learning/simple"
)

type SimpleExport struct {
	XPoints []float64
	YPoints []float64
}

type SGDExport struct {
	Thetas     []float64
	Function   string
	Iterations int
	D          int
	A          float64
	XPoints    []float64
	YPoints    []float64
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

	sort.Slice(points, func(i, j int) bool {
		return points[i].X < points[j].X
	})

	return points
}

func thetas2FunctionString(thetas ...float64) string {

	parts := make([]string, len(thetas))
	for inx, th := range thetas {
		parts[inx] = fmt.Sprint(th) + "*x^" + fmt.Sprint(inx)
	}

	return "y = " + strings.Join(parts, " + ")
}

func plotFunction(thetas ...float64) Points {

	steps := 100
	step := 1.0 / float64(steps)

	points := make(Points, steps)

	for inx := 0; inx < steps; inx++ {

		x := step * float64(inx)

		points[inx] = Point{
			X: x,
			Y: hypotheses(x, thetas...),
		}
	}

	return points
}

// hθ(x) = θ0 + θ1x1 + θ2x2
func hypotheses(x float64, thetas ...float64) float64 {

	sum := 0.0

	for idx := 0; idx < len(thetas); idx++ {
		sum += thetas[idx] * math.Pow(x, float64(idx))
	}

	return sum
}

func stochasticGradientDescent(points Points, iterations int, a float64, d int) SGDExport {

	thetas := make([]float64, d)
	for inx := 0; inx < d; inx++ {
		thetas[inx] = randFloat(-0.5, 0.5)
	}

	learnrate := make([]float64, iterations)
	for idx := 0; idx < iterations; idx++ {

		for _, point := range points {
			for j := range thetas {
				thetas[j] += a * (point.Y - hypotheses(point.X, thetas...)) * math.Pow(point.X, float64(j))
			}
		}

		jerror := 0.0

		for _, point := range points {
			jerror += math.Pow(hypotheses(point.X, thetas...)-point.Y, float64(2))
		}

		learnrate[idx] = jerror * 0.5
	}

	log.Printf(thetas2FunctionString(thetas...))
	plot := plotFunction(thetas...)
	xs, ys := plot.getXY()
	simple.WritePretty(plot.export(), "notebook/exercise.01.sgd.json")
	simple.WritePretty(learnrate, "notebook/exercise.01.learnrate.json")

	return SGDExport{
		Thetas:     thetas,
		Function:   thetas2FunctionString(thetas...),
		Iterations: iterations,
		D:          d,
		A:          a,
		XPoints:    xs,
		YPoints:    ys,
	}
}

func main() {

	// seed with birthday to get same results for every try.
	rand.Seed(28051998)

	// generate 100 random points.
	randomPoints := generateRandomPoints(100)

	// export and save generated points
	plot := randomPoints.export()
	simple.WritePretty(plot, "notebook/exercise.01.json")

	// iteration count
	iterations := 5000
	// learnRate := 0.001
	learnRate := 0.005
	// learnRate := 0.5

	data := []SGDExport{
		// stochasticGradientDescent(randomPoints, iterations, learnRate, 0),
		// stochasticGradientDescent(randomPoints, iterations, learnRate, 1),
		// stochasticGradientDescent(randomPoints, iterations, learnRate, 2),
		// stochasticGradientDescent(randomPoints, iterations, learnRate, 3),
		// stochasticGradientDescent(randomPoints, iterations, learnRate, 4),
		stochasticGradientDescent(randomPoints, iterations, learnRate, 6),
	}

	simple.WritePretty(data, "notebook/exercise.01.sgd.json")
}
