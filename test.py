from WORC import i

i.am_using_jupyter_notebooks()

i.want_to_use_images_from_this_directory('~/tmp/HN1_tutorial_subset', 'img.nii.gz', '%subject%/nii/')
i.want_to_use_segmentations_from_this_directory('~/tmp/HN1_tutorial_subset', 'mask_GTV-1.nii.gz')

i.want_to_use_labels_from_this_csv('~/tmp/HN1/labels.csv')
i.want_to_predict_this_label('Circle')

i.want_to_do_a_quick_binary_classification()
# of
i.want_to_do_a_full_binary_classification()

# fastr stdout pas printen als er een error is

# To solve this please look at the issues at github.com/WORC or create a new issue if you cannot solve it.

i.want_to_predict_this_label('Resolution')

i.want_to_do_a_quick_regression()
# of
i.want_to_do_a_full_regression()