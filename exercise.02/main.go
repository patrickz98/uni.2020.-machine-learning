package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/patrickz98/uni.2020.machine-learning/simple"
)

const (
	dataDir = "exercise.02.data"
)

type DataPoint struct {
	Dimension1 float64
	Dimension2 float64
	Label      int
}

type DataPoints []DataPoint

type Model []float64

func (model Model) Str() string {

	parts := make([]string, len(model))

	for inx, theta := range model {
		if inx == 0 {
			parts[inx] = fmt.Sprint(theta)
		} else {
			parts[inx] = fmt.Sprintf("%f * x%d", theta, inx)
		}
	}

	// return "y = " + strings.Join(parts, " + ")
	return fmt.Sprintf("y = (%f + %f * x)*(−1/(%f))", model[0], model[1], model[2])
}

func sigmoidFunction(z float64) float64 {

	return 1.0 / (1.0 + math.Pow(math.E, -z))
}

func zFunction(model Model, x1, x2 float64) float64 {

	bias := model[0]
	part1 := model[1] * x1
	part2 := model[2] * x2

	return bias + part1 + part2
}

func LogisticRegressionAlgorithm(data DataPoints) {

	model := Model{
		simple.RandFloat(-0.01, 0.01),
		simple.RandFloat(-0.01, 0.01),
		simple.RandFloat(-0.01, 0.01),
	}

	log.Printf("Init model: %v\n", model.Str())

	for count := 0; count < 100; count++ {

		for inx := range model {

			for _, point := range data {

				z := zFunction(model, point.Dimension1, point.Dimension2)
				sigmoid := sigmoidFunction(z)

				xij := 1.0

				if inx == 1 {
					xij = point.Dimension1
				}

				if inx == 2 {
					xij = point.Dimension2
				}

				model[inx] += 0.05 * (float64(point.Label) - sigmoid) * xij
			}
		}
	}

	log.Printf("Trained model: %v\n", model.Str())
}

func main() {

	rand.Seed(19980528)

	byt, err := ioutil.ReadFile(dataDir + "/data.txt")
	if err != nil {
		panic(err)
	}

	txt := string(byt)
	txt = strings.TrimSpace(txt)

	data := make(DataPoints, 0)

	for _, line := range strings.Split(txt, "\n") {

		parts := strings.Split(line, " ")

		dim1, _ := strconv.ParseFloat(parts[0], 64)
		dim2, _ := strconv.ParseFloat(parts[1], 64)
		label, _ := strconv.Atoi(parts[2])

		point := DataPoint{dim1, dim2, label}
		data = append(data, point)
	}

	LogisticRegressionAlgorithm(data)
}
