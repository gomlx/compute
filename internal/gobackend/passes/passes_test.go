package passes

import (
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}
