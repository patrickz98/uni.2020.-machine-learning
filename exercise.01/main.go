package main

import (
	"log"
	"math"
	"math/rand"
	"sort"

	"github.com/patrickz98/uni.2020.machine-learning/simple"
)

type SimpleExport struct {
	XPoints []float64
	YPoints []float64
}

type Point struct {
	X     float64
	Y     float64
	Noise float64
}

type Points []Point

func (points Points) get() (xs []float64, ys []float64) {

	xs = make([]float64, len(points))
	ys = make([]float64, len(points))

	for inx, point := range points {
		xs[inx] = point.X
		ys[inx] = point.Y
	}

	return xs, ys
}

func (points Points) export() SimpleExport {

	xi, yi := points.get()
	return SimpleExport{
		XPoints: xi,
		YPoints: yi,
	}
}

func randFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func generateRandom(num int) Points {

	points := make(Points, num)

	for inx := 0; inx < num; inx++ {

		x := rand.Float64()
		noise := randFloat(-0.3, 0.3)
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

func main() {

	rand.Seed(28051998)

	log.Println("Hallo")
	r := rand.Float64()
	log.Printf("rand: %v\n", r)

	generatePoints := 100
	points := generateRandom(generatePoints)
	plot := points.export()

	simple.WritePretty(plot, "notebook/exercise.01.json")
}
