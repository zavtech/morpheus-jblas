/**
 * Copyright (C) 2014-2017 Xavier Witdouck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.zavtech.morpheus.reference.algebra;

import java.util.Optional;
import java.util.function.Function;

import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;
import org.jblas.Singular;
import org.jblas.Solve;

import com.zavtech.morpheus.array.Array;
import com.zavtech.morpheus.frame.DataFrame;
import com.zavtech.morpheus.frame.DataFrameException;
import com.zavtech.morpheus.range.Range;
import com.zavtech.morpheus.util.LazyValue;

/**
 * An implementation of the DataFrameAlgebra interface that uses JBlas
 *
 * @param <R>   the row key type
 * @param <C>   the column key type
 *
 * <p><strong>This is open source software released under the <a href="http://www.apache.org/licenses/LICENSE-2.0">Apache 2.0 License</a></strong></p>
 *
 * @see <a href="http://jblas.org/">JBLAS</a>
 * @author  Xavier Witdouck
 */
class XDataFrameAlgebraJBlas<R,C> extends XDataFrameAlgebra<R,C> {

    /**
     * Constructor
     * @param frame     the frame reference
     */
    XDataFrameAlgebraJBlas(DataFrame<R,C> frame) {
        super(frame);
    }


    @Override
    public Decomposition decomp() {
        return new Decomp(frame());
    }


    @Override
    public DataFrame<Integer,Integer> inverse() throws DataFrameException {
        try {
            final DataFrame<R,C> frame = frame();
            final DoubleMatrix matrix = toMatrix(frame);
            if (frame().rowCount() == frame().colCount()) {
                final DoubleMatrix identity = DoubleMatrix.eye(frame.rowCount());
                final DoubleMatrix result = Solve.solve(matrix, identity);
                return toDataFrame(result);
            } else {
                final DoubleMatrix result = Solve.pinv(matrix);
                return toDataFrame(result);
            }
        } catch (Exception ex) {
            throw new DataFrameException("Failed to compute inverse of DataFrame", ex);
        }
    }


    @Override
    public DataFrame<Integer,Integer> solve(DataFrame<?,?> rhs) throws DataFrameException {
        try {
            final DataFrame<R,C> frame = frame();
            final DoubleMatrix matrix = toMatrix(frame);
            final DoubleMatrix b = toMatrix(rhs);
            if (frame().rowCount() == frame().colCount()) {
                final DoubleMatrix result = Solve.solve(matrix, b);
                return toDataFrame(result);
            } else {
                final DoubleMatrix result = Solve.solveLeastSquares(matrix, b);
                return toDataFrame(result);
            }
        } catch (Exception ex) {
            throw new DataFrameException("Failed to solve AX=B for frames", ex);
        }
    }


    /**
     * Returns a Morpheus DataFrame from the JBlas Matrix object
     * @param matrix    the JBlas matrix object
     * @return          the Morpheus DataFrame
     */
    private DataFrame<Integer,Integer> toDataFrame(DoubleMatrix matrix) {
        final Range<Integer> rowKeys = Range.of(0, matrix.rows);
        final Range<Integer> colKeys = Range.of(0, matrix.columns);
        return DataFrame.ofDoubles(rowKeys, colKeys, v -> {
            final int i = v.rowOrdinal();
            final int j = v.colOrdinal();
            return matrix.get(i, j);
        });
    }


    /**
     * Returns a JBlas DoubleMatrix representation of the Morpheus DataFrame
     * @param frame     the frame to create a JBlas matrix from
     * @return          the newly created JBlas matrix
     */
    private DoubleMatrix toMatrix(DataFrame<?,?> frame) {
        final DoubleMatrix matrix = new DoubleMatrix(frame.rowCount(), frame.colCount());
        frame.forEachValue(v -> {
            final int i = v.rowOrdinal();
            final int j = v.colOrdinal();
            final double value = v.getDouble();
            matrix.put(i, j, value);
        });
        return matrix;
    }


    /**
     * The Decomposition implementation for Apache Library
     */
    private class Decomp implements Decomposition {

        private DataFrame<?,?> frame;

        /**
         * Constructor
         * @param frame     the frame reference
         */
        Decomp(DataFrame<?,?> frame) {
            this.frame = frame;
        }

        @Override
        public <T> Optional<T> lud(Function<LUD, Optional<T>> handler) {
            return handler.apply(new XLUD(toMatrix(frame)));
        }

        @Override
        public <T> Optional<T> qrd(Function<QRD, Optional<T>> handler) {
            return handler.apply(new XQRD(toMatrix(frame)));
        }

        @Override
        public <T> Optional<T> evd(Function<EVD, Optional<T>> handler) {
            return handler.apply(new XEVD(toMatrix(frame)));
        }

        @Override
        public <T> Optional<T> svd(Function<SVD, Optional<T>> handler) {
            return handler.apply(new XSVD(toMatrix(frame)));
        }

        @Override
        public <T> Optional<T> cd(Function<CD, Optional<T>> handler) {
            return handler.apply(new XCD(toMatrix(frame)));
        }

    }


    /**
     * An implementation of LU Decomposition using the Apache Library
     */
    private class XLUD implements LUD {

        private LazyValue<DataFrame<Integer,Integer>> l;
        private LazyValue<DataFrame<Integer,Integer>> u;
        private LazyValue<DataFrame<Integer,Integer>> p;
        private Decompose.LUDecomposition<DoubleMatrix> lud;

        /**
         * Constructor
         * @param matrix    the matrix to decompose
         */
        private XLUD(DoubleMatrix matrix) {
            this.lud = Decompose.lu(matrix);
            this.l = LazyValue.of(() -> toDataFrame(lud.l));
            this.u = LazyValue.of(() -> toDataFrame(lud.u));
            this.p = LazyValue.of(() -> toDataFrame(lud.p));
        }

        @Override
        public double det() {
            return lud.u.diag().prod();
        }

        @Override
        public boolean isNonSingular() {
            throw new UnsupportedOperationException("This method is not currently supported");
        }

        @Override
        public DataFrame<Integer,Integer> getL() {
            return l.get();
        }

        @Override
        public DataFrame<Integer,Integer> getU() {
            return u.get();
        }

        @Override
        public DataFrame<Integer,Integer> getP() {
            return p.get();
        }

        @Override
        public DataFrame<Integer,Integer> solve(DataFrame<?,?> rhs) {
            throw new UnsupportedOperationException("This method is not currently supported");
        }
    }


    /**
     * An implementation of an QR Decomposition using the Apache Library
     */
    private class XQRD implements QRD {

        private LazyValue<DataFrame<Integer,Integer>> q;
        private LazyValue<DataFrame<Integer,Integer>> r;
        private Decompose.QRDecomposition<DoubleMatrix> qrd;

        /**
         * Constructor
         * @param matrix    the input matrix
         */
        XQRD(DoubleMatrix matrix) {
            this.qrd = Decompose.qr(matrix);
            this.q = LazyValue.of(() -> toDataFrame(qrd.q));
            this.r = LazyValue.of(() -> toDataFrame(qrd.r));
        }

        @Override
        public DataFrame<Integer,Integer> getR() {
            return r.get();
        }

        @Override
        public DataFrame<Integer,Integer> getQ() {
            return q.get();
        }

        @Override
        public DataFrame<Integer,Integer> solve(DataFrame<?,?> rhs) {
            throw new UnsupportedOperationException("This method is not currently supported");
        }
    }


    /**
     * An implementation of an Eigenvalue Decomposition using the Apache Library
     */
    private class XEVD implements EVD {

        private Array<Double> eigenValues;
        private DataFrame<Integer,Integer> d;
        private DataFrame<Integer,Integer> v;


        /**
         * Constructor
         * @param matrix     the input matrix
         */
        XEVD(DoubleMatrix matrix) {
            final DoubleMatrix[] vd = Eigen.symmetricEigenvectors(matrix);
            this.d = toDataFrame(vd[1]);
            this.v = toDataFrame(vd[0]);
            this.eigenValues = Array.of(Double.class, v.rowCount()).applyDoubles(v -> {
               return vd[1].get(v.index(), v.index());
            });
        }

        @Override
        public Array<Double> getEigenvalues() {
            return eigenValues;
        }

        @Override
        public DataFrame<Integer,Integer> getD() {
            return d;
        }

        @Override
        public DataFrame<Integer,Integer> getV() {
            return v;
        }
    }


    /**
     * An implementation of Singular Value Decomposition using the Apache Library
     */
    private class XSVD implements SVD {

        private int rank;
        private Array<Double> singularValues;
        private DataFrame<Integer,Integer> u;
        private DataFrame<Integer,Integer> v;
        private DataFrame<Integer,Integer> s;

        /**
         * Constructor
         * @param matrix     the input matrix
         */
        XSVD(DoubleMatrix matrix) {
            final DoubleMatrix[] usv = Singular.fullSVD(matrix);
            this.rank = computeRank(matrix, usv[1]);
            this.u = toDataFrame(usv[0]);
            this.v = toDataFrame(usv[2]);
            this.s = toDataFrame(usv[1]);
            this.singularValues = Array.of(Double.class, usv[2].rows).applyDoubles(v -> {
                final int index = v.index();
                return usv[2].get(index, index);
            });
        }

        /**
         * Computes the rank for the input matrix
         * @param a     the input matrix
         * @param s     the diagonal matrix of singular values
         * @return      the rank
         */
        private int computeRank(DoubleMatrix a, DoubleMatrix s) {
            int rank = 0;
            final int m = a.rows;
            final int n = a.columns;
            final double epsilon = Math.pow(2.0,-52.0);
            final double tolerance = Math.max(m,n) * s.get(0,0) * epsilon;
            for (int i=0; i<s.rows; i++) {
                if (s.get(i,i) > tolerance) {
                    rank++;
                }
            }
            return rank;
        }

        @Override
        public final int rank() {
            return rank;
        }

        @Override
        public final DataFrame<Integer,Integer> getU() {
            return u;
        }

        @Override
        public final DataFrame<Integer,Integer> getV() {
            return v;
        }

        @Override
        public final DataFrame<Integer,Integer> getS() {
            return s;
        }

        @Override
        public final Array<Double> getSingularValues() {
            return singularValues;
        }
    }


    /**
     * An implementation of Cholesky Decomposition using the Apache Library
     */
    private class XCD implements CD {

        private DoubleMatrix lMatrix;
        private LazyValue<DataFrame<Integer,Integer>> lFrame;

        /**
         * Constructor
         * @param matrix    the input matrix
         */
        XCD(DoubleMatrix matrix) {
            this.lMatrix = Decompose.cholesky(matrix);
            this.lFrame = LazyValue.of(() -> toDataFrame(lMatrix));
        }

        @Override
        public DataFrame<Integer,Integer> getL() {
            return lFrame.get();
        }

        @Override
        public DataFrame<Integer,Integer> solve(DataFrame<?,?> rhs) {
            throw new UnsupportedOperationException("This method is not currently supported");
        }
    }
}


