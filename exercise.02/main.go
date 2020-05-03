package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/patrickz98/uni.2020.machine-learning/simple"
)

const (
	dataDir    = "exercise.02.data"
	exportDir  = "exercise.02.notebook"
	iterations = 100
)

// Export Struct for notebook graph generation
type ExportModel struct {
	Model       Model
	ModelString string
	XPoints     []float64
	YPoints     []float64
}

// Training data point
type DataPoint struct {
	Dimension1 float64
	Dimension2 float64
	Label      int
}

type DataPoints []DataPoint

// Model: composed of three thetas (first bias)
type Model []float64

// Convert Model (thetas) in a function string
// for debug purposes.
func (model Model) str() string {
	return fmt.Sprintf("y = (%f + %f * x)*(-1/(%f))",
		model[0], model[1], model[2])
}

// Plot x and y values of Model in range from start to end.
// Once your algorithm has estimated the correct vector
// of parameters θ, you can plot such model using
// the function x2 = (θ0 + θ1x1)(−1/θ2).
func (model Model) plot(start, end float64) (xs []float64, ys []float64) {

	for start < end {
		x := start
		y := (model[0] + model[1]*x) * (-1.0 / (model[2]))

		xs = append(xs, x)
		ys = append(ys, y)

		start += 0.1
	}

	return xs, ys
}

// Convert Model for export
func (model Model) export() ExportModel {

	xs, ys := model.plot(-3, 4)

	export := ExportModel{
		Model:       model,
		ModelString: model.str(),
		XPoints:     xs,
		YPoints:     ys,
	}

	return export
}

// Sigmoid Function
func sigmoidFunction(z float64) float64 {

	return 1.0 / (1.0 + math.Pow(math.E, -z))
}

// Z function, output will be used for sigmoid function.
// Use a linear model of the form z = θ0 + θ1x1 + θ2x2.
// Don’t use a polynomial!
func zFunction(model Model, x1, x2 float64) float64 {

	bias := model[0]
	part1 := model[1] * x1
	part2 := model[2] * x2

	return bias + part1 + part2
}

// Calculate error rate (Erms)
func calculateError(data DataPoints, model Model) float64 {

	eTheta := 0.0

	for _, point := range data {
		z := zFunction(model, point.Dimension1, point.Dimension2)
		sigmoid := sigmoidFunction(z)

		eTheta += math.Pow(sigmoid-float64(point.Label), float64(2))
	}

	eTheta = eTheta * 0.5
	m := float64(len(data))
	erms := math.Sqrt((2.0 * eTheta) / m)

	return erms
}

// Logistic Regression Training Algorithm
func LogisticRegressionAlgorithm(learnRate float64, data DataPoints) {

	// (3) Initialize the parameters of your model
	// with random values in the interval (−0.01, 0.01)
	model := Model{
		simple.RandFloat(-0.01, 0.01),
		simple.RandFloat(-0.01, 0.01),
		simple.RandFloat(-0.01, 0.01),
	}

	// Log and save init Model
	log.Printf("Init model: %v\n", model.str())
	simple.WritePretty(model.export(), exportDir+"/model.init.json")

	// Error curve data
	errorCurve := make([]float64, 0)

	for count := 0; count < iterations; count++ {

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

				// modify theta
				model[inx] += learnRate * (float64(point.Label) - sigmoid) * xij
			}
		}

		// calculate error for iteration
		errorCurve = append(errorCurve, calculateError(data, model))
	}

	// Log and save final Model
	log.Printf("Trained model: %v\n", model.str())
	simple.WritePretty(model.export(), exportDir+"/model.final.json")

	// Save error curve.
	simple.WritePretty(errorCurve, exportDir+"/errorCurve.json")
}

func main() {

	rand.Seed(19980528)

	byt, err := ioutil.ReadFile(dataDir + "/data.txt")
	if err != nil {
		panic(err)
	}

	_ = os.MkdirAll(exportDir, 0755)

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

	simple.WritePretty(data, exportDir+"/trainingPoints.json")

	LogisticRegressionAlgorithm(0.1, data)
}
