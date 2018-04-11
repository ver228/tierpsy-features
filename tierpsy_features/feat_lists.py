from tierpsy_features.velocities import get_velocity_features, velocities_columns
from tierpsy_features.postures import get_morphology_features, morphology_columns, \
get_posture_features, posture_columns, posture_aux

from tierpsy_features.curvatures import get_curvature_features, curvature_columns
from tierpsy_features.food import get_cnt_feats, food_columns
from tierpsy_features.path import get_path_curvatures, path_curvature_columns, path_curvature_columns_aux

from tierpsy_features.events import event_columns


#all time series features
timeseries_feats_no_dev_columns = [velocities_columns, morphology_columns, posture_columns, \
                curvature_columns, food_columns, path_curvature_columns]
timeseries_feats_columns = sum(map(list, timeseries_feats_columns))

#add derivative columns
timeseries_feats_columns = timeseries_feats_no_dev_columns + ['d_' + x for x in timeseries_feats_no_dev_columns]

#add all the axiliary columns
aux_columns =  posture_aux + path_curvature_columns_aux
timeseries_all_columns += (timeseries_feats_columns + event_columns + aux_columns)

#cast to tuples to make data inmutable
timeseries_feats_no_dev_columns = tuple(timeseries_feats_no_dev_columns)
timeseries_feats_columns = tuple(timeseries_feats_columns)
timeseries_all_columns = tuple(timeseries_all_columns)


#add ventral features
ventral_signed_columns = [ 'relative_to_body_speed_midbody' ] 
ventral_signed_columns += path_curvature_columns + curvature_columns
ventral_signed_columns += [x for x in velocities_columns if 'angular_velocity' in x]
ventral_signed_columns += [x for x in posture_columns if 'eigen_projection' in x]
ventral_signed_columns = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]
ventral_signed_columns = tuple(ventral_signed_columns)


#all the ventral_signed_columns must be in timeseries_feats_columns
assert len(set(ventral_signed_columns) - set(timeseries_feats_columns))  == 0




