package simple

import "math/rand"

// Random float value between min and max
func RandFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
