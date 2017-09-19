# method
- segment with MOG2 (uses motion to subtract the background)
- hough line transform to get line segments
- k-means with k=2 to cluster the line segments
- calculate angle from the average of these

# assumptions
- found points will always be closer to their corresponding kmean
- bird is flying horizontally -  (ie one wing is lower than other)
- bird wings are within frame
- only one bird
