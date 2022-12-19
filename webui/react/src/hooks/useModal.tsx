import { Modal, ModalProps } from 'antd';
import React, { useState } from 'react';

type UseModalProps = Omit<Omit<ModalProps, 'open'>, 'visible'>;

const useModal = (): { Modal: React.FC<UseModalProps>; openModal: () => void } => {
  const [isOpen, setIsOpen] = useState(false);

  const Comp: React.FC<UseModalProps> = (props: UseModalProps) => (
    <Modal open={isOpen} {...props} />
  );

  return { Modal: Comp, openModal: () => setIsOpen(true) };
};

export default useModal;
