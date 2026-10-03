"""Deterministic measurement tests; synthetic geometry is not physical validation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import unittest
import numpy as np
from automation.calibration_residual_research import (
    project, summary, basis, fit_model, predict, min_line_radius, diagonal_center,
    row_column_bounds, MODELS)
from src.engine.camera.aruco_calibrator import ArucoCalibrator
import pygame


class ResidualMeasurementTests(unittest.TestCase):
    def test_camera_pixel_counts_and_sign_independent_norm(self):
        r=summary([[0,0],[1,0],[0,-2],[3,0],[0,4]])
        self.assertEqual((r['within_1'],r['within_2'],r['within_3']),(2,3,4))
        self.assertEqual(r['max'],4)

    def test_minimum_strip_bound_exact_and_rigid_invariant(self):
        p=np.array([[0.,0.],[100.,0.],[50.,10.]])
        self.assertAlmostEqual(min_line_radius(p)['radius'],5)
        angle=.71;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        self.assertAlmostEqual(min_line_radius(p@rotation+[1700,1300])['radius'],5,places=3)

    def test_straight_projected_row_has_zero_width(self):
        H=np.array([[2.,.1,1500],[.01,1.5,900],[.0001,.0002,1.]])
        p=project(H,np.c_[np.arange(0,800,100),np.full(8,500)])
        self.assertLess(min_line_radius(p)['radius'],.001)

    def test_layout_grouping_detects_bottom_row_bound(self):
        ids=[3,6,17,7,2]
        p=np.array([[0.,0.],[25,5],[50,10],[75,5],[100,0]])
        records=row_column_bounds(ids,p)
        row=next(r for r in records if r['axis']=='row_v')
        self.assertEqual(row['layout_value'],.95);self.assertGreater(row['radius'],3)

    def test_projective_center_and_corner_order(self):
        p=np.array([[0,0],[40,0],[40,40],[0,40]])
        H=np.array([[2.,.1,100],[.01,1.5,90],[.005,.001,1.]])
        observed=project(H,p);expected=project(H,[[20,20]])[0]
        np.testing.assert_allclose(diagonal_center(observed),expected,atol=1e-10)
        np.testing.assert_allclose(diagonal_center(np.roll(observed,1,axis=0)),expected,atol=1e-10)
        self.assertGreater(np.linalg.norm(observed.mean(0)-expected),.1)

    def test_fixed_models_recover_exact_projective_data_without_mutation(self):
        viewport=[20,10,760,580];cal=ArucoCalibrator(pygame.Rect(viewport))
        client=np.array(list(cal.rendered_marker_positions().values()))
        matrix=np.array([[2.,.1,1500],[.01,1.5,900],[.0001,.0002,1.]])
        camera=project(matrix,client);a=client.copy();b=camera.copy()
        for name in MODELS:
            m=fit_model(name,client,camera,viewport)
            np.testing.assert_allclose(predict(m,client),camera,atol=.001)
            self.assertFalse(m['live_installable'])
            if name=='QUADRATIC_RESIDUAL':self.assertNotIn('camera_to_client',m)
            else:np.testing.assert_allclose(project(m['camera_to_client'],predict(m,client)),client,atol=1e-9)
        np.testing.assert_array_equal(client,a);np.testing.assert_array_equal(camera,b)

    def test_basis_respects_nonzero_viewport_origin(self):
        a=basis([[20,10],[780,590]],[20,10,760,580])
        b=basis([[120,60],[880,640]],[120,60,760,580])
        np.testing.assert_array_equal(a,b)

    def test_holdout_fits_only_supplied_points(self):
        viewport=[20,10,760,580];c=ArucoCalibrator(pygame.Rect(viewport))
        x=np.array(list(c.rendered_marker_positions().values()));y=project(np.array([[2,0,1000],[0,2,500],[0,0,1.]]),x)
        m=fit_model('QUADRATIC_RESIDUAL',x[:-1],y[:-1],viewport)
        expected=predict(m,x[-1:]);y[-1]+=[100,100]
        m2=fit_model('QUADRATIC_RESIDUAL',x[:-1],y[:-1],viewport)
        np.testing.assert_array_equal(predict(m2,x[-1:]),expected)


if __name__=='__main__':unittest.main()
