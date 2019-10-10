import tensorflow as tf

def dice_score(predictions, labels, num_classes, session=None):
    # Dice is (2TP) / (2TP + FP + FN)
    #tp, _ = streaming_true_positives(predictions, labels)
    #fp, _ = streaming_false_positives(predictions, labels)
    #fn, _ = streaming_false_negatives(predictions, labels)
    predictions = tf.to_int32(tf.round(predictions))
    labels = tf.to_int32(tf.round(labels))

    #pred_run = session.run(predictions)
    #labels_run = session.run(labels)
    dice_sum = tf.zeros([], dtype=tf.float32)
    dice_num = tf.zeros([], dtype=tf.float32)
    dice_scores = []

    for class_val in range(num_classes - 1):


        #This is an ugly hack for the cityscapes dataset - we only evaluate on ids which are used for evaluation
        if num_classes > 2:
            if (class_val + 1) not in [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]:
                continue

        #Check if label even contains the given class



        ones_like_actuals = tf.add(tf.ones_like(labels), (class_val))
        zeros_like_actuals = tf.zeros_like(labels)
        ones_like_predictions = tf.add(tf.ones_like(predictions), (class_val))
        zeros_like_predictions = tf.zeros_like(predictions)

        #ones_actuals_run = session.run(ones_like_actuals)
        #zeros_actuals_run = session.run(zeros_like_actuals)
        #ones_preds_run = session.run(ones_like_predictions)
        #zeros_preds_run = session.run(zeros_like_predictions)

        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.not_equal(labels, ones_like_predictions),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, ones_like_actuals),
                    tf.not_equal(predictions, ones_like_actuals)
                ),
                "float"
            )
        )


        #tp_op_run = session.run(tp_op)
        #fp_op_run = session.run(fp_op)
        #fn_op_run = session.run(fn_op)





        double_tp = tf.multiply(tp_op, 2)

        dice_score_final = double_tp / (tf.add_n([double_tp, fp_op, fn_op]) +  0.0000001)


        #dice_sum_add = tf.cond(tf.greater_equal(tp_op + fn_op, 0.01), lambda: dice_score_final, lambda: tf.zeros([], dtype=tf.float32))
        #dice_num_add = tf.cond(tf.greater_equal(tp_op + fn_op, 0.01), lambda: tf.ones([], dtype=tf.float32), lambda: tf.zeros([], dtype=tf.float32))
        #dice_sum = dice_sum + dice_sum_add
        #dice_num = dice_num + dice_num_add
        #test_cond = session.run(tf.greater_equal(tp_op + fn_op, 0.01))
        #test_add = session.run(dice_sum_add)
        #test_sum = session.run(dice_sum)
        #test_num = session.run(dice_num)
        #a = 3

        dice_scores.append(dice_score_final)
        #dice_scores.append(tf.add_n([double_tp, fp_op, fn_op]))

        #return  double_tp / (tf.add_n([double_tp, fp_op, fn_op]))

    return tf.reduce_mean(dice_scores)