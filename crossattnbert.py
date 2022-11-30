class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        # initilize word embedding
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens

        # initlize the vision embedding
        self.v_embeddings = BertImageEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.apply(self.init_weights)

    def forward(
        self,
        input_aud,
        input_imgs,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        task_ids=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_aud)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_aud)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_aud.size(0), input_imgs.size(1), input_aud.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        #embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        #v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        embedding_output=input_aud
        v_embedding_ouutput=input_imgs
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )

class BertForMultiModalPreTraining(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self, config):
        super(BertForMultiModalPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )

        self.apply(self.init_weights)
        self.visual_target = config.visual_target
        self.num_negative = config.num_negative
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

        print("model's visual target is ", config.visual_target)

        if self.visual_target == 0:
            self.vis_criterion = nn.KLDivLoss(reduction="none")
        elif self.visual_target == 1:
            self.vis_criterion = nn.MSELoss(reduction="none")
        elif self.visual_target == 2:
            self.vis_criterion = CrossEntropyLoss()

        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
        output_all_attention_masks=False,
    ):
        # in this model, we first embed the images.
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        if (
            masked_lm_labels is not None
            and next_sentence_label is not None
            and image_target is not None
        ):
            prediction_scores_v = prediction_scores_v[:, 1:]
            if self.visual_target == 1:
                img_loss = self.vis_criterion(prediction_scores_v, image_target)
                masked_img_loss = torch.sum(
                    img_loss * (image_label == 1).unsqueeze(2).float()
                ) / max(
                    torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1
                )

            elif self.visual_target == 0:
                img_loss = self.vis_criterion(
                    F.log_softmax(prediction_scores_v, dim=2), image_target
                )

                masked_img_loss = torch.sum(
                    img_loss * (image_label == 1).unsqueeze(2).float()
                ) / max(torch.sum((image_label == 1)), 0)
            elif self.visual_target == 2:
                # generate negative sampled index.
                num_negative = self.num_negative
                num_across_batch = int(self.num_negative * 0.7)
                num_inside_batch = int(self.num_negative * 0.3)

                batch_size, num_regions, _ = prediction_scores_v.size()
                assert batch_size != 0
                # random negative across batches.
                row_across_index = input_ids.new(
                    batch_size, num_regions, num_across_batch
                ).random_(0, batch_size - 1)
                col_across_index = input_ids.new(
                    batch_size, num_regions, num_across_batch
                ).random_(0, num_regions)

                for i in range(batch_size - 1):
                    row_across_index[i][row_across_index[i] == i] = batch_size - 1
                final_across_index = row_across_index * num_regions + col_across_index

                # random negative inside batches.
                row_inside_index = input_ids.new(
                    batch_size, num_regions, num_inside_batch
                ).zero_()
                col_inside_index = input_ids.new(
                    batch_size, num_regions, num_inside_batch
                ).random_(0, num_regions - 1)

                for i in range(batch_size):
                    row_inside_index[i] = i
                for i in range(num_regions - 1):
                    col_inside_index[:, i, :][col_inside_index[:, i, :] == i] = (
                        num_regions - 1
                    )
                final_inside_index = row_inside_index * num_regions + col_inside_index

                final_index = torch.cat((final_across_index, final_inside_index), dim=2)

                # Let's first sample where we need to compute.
                predict_v = prediction_scores_v[image_label == 1]
                neg_index_v = final_index[image_label == 1]

                flat_image_target = image_target.view(batch_size * num_regions, -1)
                # we also need to append the target feature at the begining.
                negative_v = flat_image_target[neg_index_v]
                positive_v = image_target[image_label == 1]
                sample_v = torch.cat((positive_v.unsqueeze(1), negative_v), dim=1)

                # calculate the loss.
                score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
                masked_img_loss = self.vis_criterion(
                    score, input_ids.new(score.size(0)).zero_()
                )

            # masked_img_loss = torch.sum(img_loss) / (img_loss.shape[0] * img_loss.shape[1])
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )

            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            return (
                masked_lm_loss.unsqueeze(0),
                masked_img_loss.unsqueeze(0),
                next_sentence_loss.unsqueeze(0),
            )
        else:
            return (
                prediction_scores_t,
                prediction_scores_v,
                seq_relationship_score,
                all_attention_mask,
            )
