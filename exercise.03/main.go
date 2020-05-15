package main

import (
	"fmt"
	"image/png"
	"io/ioutil"
	"os"
	"strings"
)

const (
	dataDir = "exercise.03.data"
)

type FloatMatrix [][]float64

func (matrix FloatMatrix) Add(matrix2 FloatMatrix) FloatMatrix {

	values := make(FloatMatrix, len(matrix))

	for inx := 0; inx < len(matrix); inx++ {

		values[inx] = matrix[inx]

		for iny := 0; iny < len(matrix[inx]); iny++ {
			values[inx][iny] += matrix2[inx][iny]
		}
	}

	return values
}

type FloatMatrixs []FloatMatrix

func (matrixs FloatMatrixs) Mean() FloatMatrix {

	values := make(FloatMatrix, 0)

	for _, matrix := range matrixs {

		if len(values) == 0 {
			values = make([][]float64, len(matrix))
		}

		for inx, row := range matrix {

			if len(values[inx]) == 0 {
				values[inx] = make([]float64, len(row))
			}

			for iny, val := range row {
				values[inx][iny] += val
			}
		}
	}

	for inx := range values {
		for iny, val := range values[inx] {
			values[inx][iny] = val / float64(len(matrixs))
		}
	}

	return values
}

type FloatVector []float64

func (vector FloatVector) Subtract(vector2 FloatVector) FloatVector {

	values := make(FloatVector, 0)

	for inx := 0; inx < len(vector); inx++ {
		values = append(values, vector[inx]-vector2[inx])
	}

	return values
}

func (vector FloatVector) MatrixProduct(vector2 FloatVector) FloatMatrix {

	matrix := make(FloatMatrix, len(vector))

	for inx := 0; inx < len(vector); inx++ {

		matrix[inx] = make([]float64, len(vector2))

		for iny := 0; iny < len(vector2); iny++ {
			matrix[inx][iny] = vector[inx] * vector2[iny]
		}
	}

	return matrix
}

type FeatureVector struct {
	RedMin   int
	GreenMin int
	BlueMin  int
	RedAvg   float64
	GreenAvg float64
	BlueAvg  float64
}

func (vector FeatureVector) Values() FloatVector {

	return []float64{
		float64(vector.RedMin),
		float64(vector.GreenMin),
		float64(vector.BlueMin),
		vector.RedAvg,
		vector.GreenAvg,
		vector.BlueAvg,
	}
}

type FeatureVectors []FeatureVector

func (vectors FeatureVectors) Mean() FeatureVector {

	mean := FeatureVector{}

	for _, vector := range vectors {
		mean.RedMin += vector.RedMin
		mean.GreenMin += vector.GreenMin
		mean.BlueMin += vector.BlueMin

		mean.RedAvg += vector.RedAvg
		mean.GreenAvg += vector.GreenAvg
		mean.BlueAvg += vector.BlueAvg
	}

	return FeatureVector{
		RedMin:   mean.RedMin / len(vectors),
		GreenMin: mean.GreenMin / len(vectors),
		BlueMin:  mean.BlueMin / len(vectors),
		RedAvg:   mean.RedAvg / float64(len(vectors)),
		GreenAvg: mean.GreenAvg / float64(len(vectors)),
		BlueAvg:  mean.BlueAvg / float64(len(vectors)),
	}
}

type Pixel struct {
	Red   int
	Green int
	Blue  int
	Alpha int
}

type Pixels []Pixel

func (pixels Pixels) ToVector() FeatureVector {

	rmin := 0xff
	gmin := 0xff
	bmin := 0xff

	for _, pixel := range pixels {
		if rmin > pixel.Red {
			rmin = pixel.Red
		}

		if gmin > pixel.Green {
			gmin = pixel.Green
		}

		if bmin > pixel.Blue {
			bmin = pixel.Blue
		}
	}

	rsum := 0
	gsum := 0
	bsum := 0

	for _, pixel := range pixels {
		rsum += pixel.Red
		gsum += pixel.Green
		bsum += pixel.Blue
	}

	rAvg := float64(rsum) / float64(len(pixels))
	gAvg := float64(gsum) / float64(len(pixels))
	bAvg := float64(bsum) / float64(len(pixels))

	return FeatureVector{
		RedMin:   rmin,
		GreenMin: gmin,
		BlueMin:  bmin,
		RedAvg:   rAvg,
		GreenAvg: gAvg,
		BlueAvg:  bAvg,
	}
}

// func newPixel(color byte) Pixel {
// 	alpha := color & 0xff
// 	blue  := (color >> 8) & 0xff
// 	green := (color >> 16) & 0xff
// 	red   := (color >> 24) & 0xff
//
// 	return Pixel{
// 		Red:   red,
// 		Green: green,
// 		Blue:  blue,
// 		Alpha: alpha,
// 	}
// }

func readFilePixels(filename string) Pixels {

	infile, _ := os.Open(filename)
	defer func() {
		_ = infile.Close()
	}()

	img, err := png.Decode(infile)
	if err != nil {
		panic(err)
	}

	pixels := make(Pixels, 0)

	size := img.Bounds().Max

	for inx := 0; inx < size.X; inx++ {
		for iny := 0; iny < size.Y; iny++ {
			color := img.At(inx, iny)
			r, g, b, a := color.RGBA()
			r, g, b, a = r>>8, g>>8, b>>8, a>>8

			pixel := Pixel{
				Red:   int(r),
				Green: int(g),
				Blue:  int(b),
				Alpha: int(a),
			}

			pixels = append(pixels, pixel)
		}
	}

	return pixels
}

func readExamples(name string) FeatureVectors {

	featureVectors := make(FeatureVectors, 0)

	source := dataDir + "/" + name
	files, _ := ioutil.ReadDir(source)

	for _, file := range files {

		if !strings.HasSuffix(file.Name(), ".png") {
			continue
		}

		filename := source + "/" + file.Name()

		fmt.Println(filename)

		pixels := readFilePixels(filename)
		featureVectors = append(featureVectors, pixels.ToVector())
	}

	return featureVectors
}

func GaussianDiscriminantAnalysis(positives, negatives FeatureVectors) {

	positivesLen := len(positives)
	negativesLen := len(negatives)
	phi := float64(positivesLen) / float64(positivesLen+negativesLen)
	fmt.Println("phi", phi)

	positivesMean := positives.Mean()
	fmt.Println("positivesMean", positivesMean)

	negativesMean := negatives.Mean()
	fmt.Println("negativesMean", negativesMean)

	coffSum := make(FloatMatrixs, 0)

	for _, feature := range positives {
		vectorX := feature.Values()
		vectorY := positivesMean.Values()

		xxx := vectorX.Subtract(vectorY)
		coffSum = append(coffSum, xxx.MatrixProduct(xxx))
	}

	for _, feature := range negatives {
		vectorX := feature.Values()
		vectorY := negativesMean.Values()

		xxx := vectorX.Subtract(vectorY)
		coffSum = append(coffSum, xxx.MatrixProduct(xxx))
	}

	coff := coffSum.Mean()
	fmt.Println("coff", coff)
}

func main() {

	negatives := readExamples("negatives")
	positives := readExamples("positives")

	fmt.Println(negatives)
	fmt.Println(positives)

	GaussianDiscriminantAnalysis(positives, negatives)
}
