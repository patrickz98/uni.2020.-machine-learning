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

type FeatureVector struct {
	RedMin   int
	GreenMin int
	BlueMin  int
	RedAvg   float64
	GreenAvg float64
	BlueAvg  float64
}

type FeatureVectors []FeatureVector

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
		if rmin < pixel.Red {
			rmin = pixel.Red
		}

		if gmin < pixel.Green {
			gmin = pixel.Green
		}

		if bmin < pixel.Blue {
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

func main() {

	negatives := readExamples("negatives")
	positives := readExamples("positives")

	fmt.Println(negatives)
	fmt.Println(positives)
}
