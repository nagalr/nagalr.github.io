---
layout: post
title:  "Merge Sort Implementation in C#"
date:   2009-09-05 15:54:28 +0500
categories: [code, programming, C#, algorithm]
tags: [code, programming, C#, algorithm]
---
# Introduction
Sorting is a common activity in the programming work. Merge-Sort is one of the best implementation for sorting, since its running on O(nlogn) - which is the best run-time available for sorting.

On one of my projects I had a requirement to sort Objects. After short research, I decided to write a Generic Merge-Sort Class that will work for me. During the work, I found a few major points that should be considered while using that Generic Class for Sorting.

<img class="img-fluid" src="/assets/img/posts/MergeSort.png"/>

# Background
Merge Sort (pseudo-code in the picture) is a recursive algorithm, that splits an array of the objects to 2 sub arrays - A,B (which initialize to infinite values), sorts this into 2 sub-arrays and finally merges them. (Hence, 'merge' sort)

# Using the Code
For using the Generic Merge Sort, you should (probably) know what is the objects parameter by which the objects will be sorted.

Now, since the 2 sub-arrays should be initialized with an infinite value, you should use the default-ctor to make that initialization. Example: For sorting an array of 'Persons' Objects that will be sorted according to their ID number (from low to high), you should declare infinite ID-number (infinite int32 value) in the default-ctor. (I know This is not an elegant implementation, but In that case there is no alternative.)

{% highlight C# %}
class MergeSort
    {
        /// <span class="code-SummaryComment"><summary></span>
        /// Sorts an array of Objects
        /// IComparable - use 'CompareTo' to compare objects
        /// where T : new() - need to create a new Type 'T' object inside the method
        /// <span class="code-SummaryComment"></summary></span>
        /// <span class="code-SummaryComment"><param name="X"></param></span>
        public static T[] Merge_Sort<T>(T[] X) where T : IComparable, new()
        {
            int n = X.Length;
            X = MegrgeSort_Internal(X, n);
            return X;
        }

        /// <span class="code-SummaryComment"><summary></span>

        /// Internal method for sorting
        /// <span class="code-SummaryComment"></summary></span>
        /// <span class="code-SummaryComment"><param name="X"></param></span>
        /// <span class="code-SummaryComment"><param name="n"></param></span>
        private static T[] MegrgeSort_Internal<T>(T[] X, int n) where T : IComparable,
            new() 
        {
            // Define 2 aid Sub-Arrays
            T[] A = new T[(n / 2) + 2];
            T[] B = new T[(n / 2) + 2];

            // Initialize the 2 Sub-Arrays with an infinite 'sorting parameter'
            // Therefor, You must include a default ctor in your class
            // which will initialize an infinite value - To the sorting parameter
            // using 'where T : new()' here
            for (int i = 0; i < A.Length; i++)
            {
                A[i] = new T(); ;
                B[i] = new T();
            }

            // Recursive Stop-Condition, Sorting a Basic Array (Size 2)
            if (n == 2)
            {
                int CompareValue = X[0].CompareTo(X[1]);
                if (CompareValue > 0)
                {
                    T tempT = X[0];
                    X[0]    = X[1];
                    X[1]    = tempT;
                }
            }

            else
            {
                // The Sub-Arrays Size is Large than 2
                if (n > 2)
                {
                    int m = n / 2;

                    // Initialize the 2 Sub-Arrays (The first relevant values)
                    for (int i = 0; i < m; i = i + 1)
                    {
                        A[i] = X[i];
                    }

                    for (int j = m; j < n; j++)
                    {
                        B[j - m] = X[j];
                    }

                    // 2 Recursive Calling, Sorting Sub-Arrays
                    A = MegrgeSort_Internal(A, m);
                    B = MegrgeSort_Internal(B, n - m);

                    // Merging the Sorted Sub-Arrays into the main Array
                    int p = 0;
                    int q = 0;

                    for (int k = 0; k < n; k++)
                    {
                        {
                            int CompareValure = A[p].CompareTo(B[q]);

                            if (CompareValure == 0 ||
                                CompareValure == -1)
                            {
                                X[k] = A[p];
                                p = p + 1;
                            }

                            else
                            {
                                X[k] = B[q];
                                q = q + 1;
                            }
                        }
                    }

                } // if

            } // else

            return X;

        } // MegrgeSort_Internal
    }
{% endhighlight %}

# Points of Interest
When writing a class for using Generic Merge Sort, you should implement the IComparable interface since the code using the CompareTo function - to compare objects. You should also write your own 'overloading operators' that will be in use to compare objects: ==, !=, >, >=, <, <= plus Equals and GetHashCode.

Please notice that GetHashCode will be in use by the CLR implicitly - so you must write your own implementation that will return the same value for the same object.

There are two projects attached:

The first one is a simple (int32) Implementation, that sorts an array of integers. The second one is the Generic Merge Sort, Plus a simple 'Person' class that will demonstrate the using of the Generic code.

# License
This article, along with any associated source code and files, is licensed under The Code Project Open License (CPOL)